!function() { 

   // module container
   var sequences = {};

   sequences.pointParser = pointParser;
   function pointParser(serves) {
      var code = serves.join('|');
      // parse first serve in case it ends the point
      // sometimes there is erroneous 2nd serve data
      var s1result = shotParser(serves[0], 1);

      if (s1result.winner == 'S' || !serves[1] ) {
         s1result.serve = 1;
         s1result.code = code;
         return s1result;
      }

      s2result = shotParser(serves[1], 2);
      s2result.serve = 2;
      s2result.first_serve = { serves: s1result.serves, }
      if (s1result.lets) s2result.first_serve.lets = s1result.lets;
      if (s1result.error) s2result.first_serve.error = s1result.error;
      if (s1result.parse_notes) s2result.first_serve.parse_notes = s1result.parse_notes;

      s2result.code = code;
      return s2result;
   }

   sequences.shotParser = shotParser;
   function shotParser(shot_sequence, which_serve) {

      var point;
      var parsed_shots = analyzeSequence(shot_sequence);

      if (['Q', 'S'].indexOf(parsed_shots.result) >= 0) {
         parsed_shots.winner = 'S';
         return parsed_shots;
      }

      if (['P', 'R'].indexOf(parsed_shots.result) >= 0) {
         parsed_shots.winner = 'R';
         return parsed_shots;
      }

      // if there is not a terminator for second serve Receiver is always the winner
      if (!parsed_shots.terminator) {
         if (!shotFault(parsed_shots.serves[0]) && parsed_shots.serves.length > 2) {
            parsed_shots.winner = 'R'; 
            return parsed_shots;
         }
      }

      // even number of shots implies Receiver made final shot
      var last_player = (parsed_shots.serves.length + parsed_shots.rally.length) % 2 == 0 ? 'R' : 'S';
      var final_shot = parsed_shots.rally.length ? parsed_shots.rally[parsed_shots.rally.length - 1] : parsed_shots.serves[parsed_shots.serves.length - 1];

      // if there is no shot in the sequence, continue to next shot_sequence
      if (!final_shot) { 
         return { continue: true };
      }

      // if there is no rally
      if (!parsed_shots.rally.length) {
         if (parsed_shots.terminator == '*') { 
            parsed_shots.result = 'Ace'; 
            parsed_shots.winner = 'S';
         } else if (parsed_shots.terminator == '#') { 
            parsed_shots.result = 'Serve Winner'; 
            parsed_shots.winner = 'S';
         } else if (shotFault(parsed_shots.serves[0])) {
            parsed_shots.error = assignError(parsed_shots.serves[0]);
            if (which_serve == 2) {
               parsed_shots.result = 'Double Fault';
               parsed_shots.winner = 'R';
            }
         } else {
            parsed_shots.parse_notes = 'treated as a fault';
            if (which_serve == 2) {
               parsed_shots.result = 'Double Fault';
               parsed_shots.winner = 'R';
            }
         }
         return parsed_shots;
      }

      if (final_shot.indexOf('#') >= 0) {
         parsed_shots.result = 'Forced Error';
         parsed_shots.error = assignError(final_shot);
         if (!shotFault(parsed_shots.serves[0])) {
            parsed_shots.winner = (last_player == 'R') ? 'S' : 'R';
         } else {
            // doesn't make sense, but this is how the spreadsheet does it...
            parsed_shots.winner = 'S';
         }
      } else if (final_shot.indexOf('*') >= 0) {
         parsed_shots.result = 'Winner';
         parsed_shots.winner = last_player;
      } else if (final_shot.indexOf('@') >= 0) {
         parsed_shots.result = 'Unforced Error';
         parsed_shots.error = assignError(final_shot);
         if (!shotFault(parsed_shots.serves[0])) {
            parsed_shots.winner = (last_player == 'R') ? 'S' : 'R';
         } else {
            // doesn't make sense, but this is how the spreadsheet does it...
            parsed_shots.winner = 'R';
         }
      } else if (!shotFault(parsed_shots.serves[0])) {
         if (parsed_shots.serves.length && parsed_shots.rally.length) {
            parsed_shots.parse_notes = 'no terminator: receiver wins point';
            parsed_shots.result = 'Unknown';
            parsed_shots.winner = 'R';
         } else if (parsed_shots.rally.length == 1 && shotFault(final_shot)) {
            parsed_shots.error = assignError(final_shot);
            parsed_shots.winner = (last_player == 'R') ? 'S' : 'R';
         }
      } else if (parsed_shots.rally.length == 1 && shotFault(final_shot)) {
         parsed_shots.error = assignError(final_shot);
         parsed_shots.winner = (last_player == 'R') ? 'S' : 'R';
      }

      return parsed_shots;
   }

   function assignError(shot) {
      var errors = {'n': 'Net', 'w': 'Out Wide', 'd': 'Out Long', 'x': 'Out Wide and Long', 'g': 'Foot Fault', 'e': 'Unknown', '!': 'Shank' };
      var error = shotFault(shot);
      if (error) return errors[error];
   }

   sequences.analyzeSequence = analyzeSequence;
   function analyzeSequence(shot_sequence) {
      var result;
      var terminator;
      var ignored_shots;

      // count lets
      var lets = shot_sequence.split('c').length - 1;
      // remove all lets
      shot_sequence = shot_sequence.split('c').join('');

      var shots = shotSplitter(shot_sequence);
      var trimmed_shots = shots;

      // eliminate any sequence data following terminator
      for (var s = shots.length - 1; s>=0; s--) {
         terminator = containsTerminator(shots[s]);
         if (terminator) {
            trimmed_shots = shots.slice(0, s + 1);
            ignored_shots = shots.slice(s + 1);
            result = shots[s];
            break;
         }
      }
      var serves = findServes(trimmed_shots);
      var rally = serves.length ? trimmed_shots.slice(serves.length) : trimmed_shots;

      if (!terminator && !serves.length && rally.length == 1 && ['S', 'P', 'Q', 'R'].indexOf(rally[0]) >= 0) {
         result = rally[0];
         rally = [];
      }
      var analysis = { serves: serves, rally: rally };
      if (lets) analysis.lets = lets;
      if (terminator) analysis.terminator = terminator;
      if (result) analysis.result = result;
      if (ignored_shots && ignored_shots.length) analysis.ignored = ignored_shots;

      return analysis
   }

   sequences.decipherPoint = decipherPoint;
   function decipherPoint(point) {
      if (!point || !point.serves || !point.rally) return false;
      var sequence = point.serves.join('') + point.rally.join('');
      return decipherSequence(sequence, point.point);
   }

   sequences.decipherSequence = decipherSequence;
   function decipherSequence(sequence, point) {
      var description = [];;
      var shots = shotSplitter(sequence);
      var origin = 0;
      var incoming_direction = 0;
      if (point) {
         if (deuce_court_points.indexOf(point) >= 0) {
            origin = 1;
            incoming_direction = 1;
         } else if (ad_court_points.indexOf(point) >= 0) {
            origin = 3;
            incoming_direction = 3;
         }
      }
      shots.forEach(function(shot) {
         var analysis = decipherShot(shot, point, incoming_direction, origin);
         description.push(analysis.sequence);
         incoming_direction = analysis.direction;
      });
      return description;
   }

   sequences.decipherShot = decipherShot;
   function decipherShot(shot, point, incoming_direction, origin) {
      // point is needed to determine side from which serve originated
      // incoming_direction is the direction of the previous shot
      // need to add player position from previous shot
      // so that inside-in and inside-out can be properly calculated
     
      var assignments = {
         'S': 'Server won the point',
         'P': 'Penalty against Server',
         'Q': 'Penalty against Receiver',
         'R': 'Receiver won the point'
      };
      var errors = {
         'n': 'Netted',
         'w': 'Out Wide',
         'd': 'Out Long',
         'x': 'Out Wide and Long',
         'g': 'Foot Fault',
         'e': 'Unknown Error',
         '!': 'Shank' 
      };
      var serves = {
         '0':  'Unknown Serve',
         '4':  'Wide Serve',
         '5':  'Body Serve',
         '6':  'T Serve'
      };
      var forehand = {
         'f':  'Forehand',
         'r':  'Forehand Slice',
         'v':  'Forehand Volley',
         'o':  'Overhead Smash',
         'u':  'Forehand Drop Shot',
         'l':  'Forehand Lob',
         'h':  'Forehand Half-volley',
         'j':  'Forehand Drive Volley'
      };
      var backhand = {
         'b':  'Backhand',
         's':  'Backhand Slice',
         'z':  'Backhand Volley',
         'p':  'Backhand Overhead Smash',
         'y':  'Backhand Drop Shot',
         'm':  'Backhand Lob',
         'i':  'Backhand Half-volley',
         'k':  'Backhand Drive Volley'
      };
      var other = {
         't':  'Trick Shot',
         'q':  'Unknown Shot'
      };
      var directions = {
         '1':  'to Right Hander Forehand',
         '2':  'Down the Middle',
         '3':  'to Right Hander Backhand'
      };
      var depths = {
         '7':  '(shallow)',
         '8':  '(deep)',
         '9':  '(very deep)'
      };
      var terminators = {
         '*':  'Winner',
         '#':  'Forced Error',
         '@':  'Unforced Error'
      };
      var positions = {
         '+': 'approach shot',
         '-': 'at the Net',
         '=': 'at the Baseline'
      };

      var incidentals = {
         ';': '(net cord)',
         'c': '(Let)'
      }

      var strokes = {};
      Object.keys(forehand).forEach(function(e) { strokes[e] = forehand[e] });
      Object.keys(backhand).forEach(function(e) { strokes[e] = backhand[e] });
      Object.keys(other).forEach(function(e) { strokes[e] = other[e] });

      // create aggregate object
      var babel = {};
      Object.keys(errors).forEach(function(e) { babel[e] = errors[e] });
      Object.keys(serves).forEach(function(e) { babel[e] = serves[e] });
      Object.keys(strokes).forEach(function(e) { babel[e] = strokes[e] });
      Object.keys(directions).forEach(function(e) { babel[e] = directions[e] });
      Object.keys(depths).forEach(function(e) { babel[e] = depths[e] });
      Object.keys(terminators).forEach(function(e) { babel[e] = terminators[e] });
      Object.keys(positions).forEach(function(e) { babel[e] = positions[e] });
      Object.keys(incidentals).forEach(function(e) { babel[e] = incidentals[e] });

      // create sequence string from shot as entered by coder
      var full_sequence = shot.split('').map(function(m) {
         return babel[m] ? babel[m] + ', ' : ''
      }).join('');
      full_sequence = full_sequence.length > 2 ? full_sequence.slice(0, full_sequence.length - 2) : '';

      // break shot down into descriptive elements
      var sequence;
      var direction;
      var stroke = shot[0];
      if (assignments[stroke]) {
         sequence = assignments[stroke];
      } else if (serves[stroke]) {
         sequence = serves[stroke];
         if (shot.indexOf('+') > 0) sequence += '; net approach';
         var fault = shotFault(shot);
         if (fault) sequence += '; ' + errors[fault];
         var terminator = containsTerminator(shot);
         if (terminator) {
            if (terminator == '*') sequence += '; Ace';
            if (terminator == '#') sequence += '; Serve Winner';
         }
         if (point) {
            if (stroke == 5) {
               direction = 2;
            } else if (deuce_court_points.indexOf(point) >= 0) {
               direction = 1;
            } else if (ad_court_points.indexOf(point) >= 0) {
               direction = 3;
            }
         } else {
            direction = 0;
         }
      // need to add position information as well as incoming direction
      } else if (strokes[stroke]) {
         var hand;
         if (forehand[stroke]) hand = 'forehand';
         if (backhand[stroke]) hand = 'backhand';
         sequence = strokes[stroke];
         var position = shotPosition(shot);
         if (position) sequence += ' ' + positions[position];
         var incidental = shotIncidental(shot);
         if (incidental) sequence += ' ' + incidentals[incidental];
         var direction = shotDirection(shot);
         if (direction) {
            if (direction == 1 && incoming_direction == 1 || direction == 3 && incoming_direction == 3) {
               sequence += ' crosscourt';
            } else if (direction == 3 && incoming_direction == 1 || direction == 1 && incoming_direction == 3) {
               sequence += ' down the line';
            } else if (direction == 2 && incoming_direction == 2) {
               sequence += ' down the middle';
            } else if (direction == 2) {
               sequence += ' to the middle';
            } else if (direction == 3) {
               sequence += ' to the right side';
            } else if (direction == 1) {
               sequence += ' to the left side';
            }
         }
         var depth = shotDepth(shot);
         var shot_depth = depth ? depths[depth] : '';
         if (shot_depth) sequence += '; ' + shot_depth;

         var fault = shotFault(shot);
         if (fault) sequence += '; ' + errors[fault];
         var terminator = containsTerminator(shot);
         if (terminator) sequence += '; ' + terminators[terminator];
      } else {
         var fault = shotFault(shot);
         if (fault) sequence = errors[fault];
      }

      return { sequence: sequence, full_sequence: full_sequence, direction: direction };
   }

   function shotIncidental(shot) {
      if (!shot) return false;
      var incidentals = ';'.split('');
      for (var d=0; d < incidentals.length; d++) {
         if (shot.indexOf(incidentals[d]) >= 0) return incidentals[d];
      }
      return false;
   }

   function shotPosition(shot) {
      if (!shot) return false;
      var positions = '+-='.split('');
      for (var d=0; d < positions.length; d++) {
         if (shot.indexOf(positions[d]) >= 0) return positions[d];
      }
      return false;
   }

   function shotDepth(shot) {
      if (!shot) return false;
      var depths = '789'.split('');
      for (var d=0; d < depths.length; d++) {
         if (shot.indexOf(depths[d]) >= 0) return depths[d];
      }
      return false;
   }

   function shotDirection(shot) {
      if (!shot) return false;
      var directions = '123'.split('');
      for (var d=0; d < directions.length; d++) {
         if (shot.indexOf(directions[d]) >= 0) return directions[d];
      }
      return false;
   }

   sequences.shotFault = shotFault;
   function shotFault(shot) {
      if (!shot) return false;
      var faults = 'nwdxge!'.split('');
      for (var f=0; f < faults.length; f++) {
         if (shot.indexOf(faults[f]) >= 0) return faults[f];
      }
      return false;
   }

   function shotType(shot) {
      if (!shot) return false;
      var shot_types = '0456fbrsvzopuylmhijktq'.split('');
      for (var f=0; f < shot_types.length; f++) {
         if (shot.indexOf(shot_types[f]) >= 0) return shot_types[f];
      }
      return false;
   }

   sequences.containsTerminator = containsTerminator;
   function containsTerminator(shot) {
      if (!shot) return false;
      var terminators = ['#', '@', '*'];
      for (var t=0; t < terminators.length; t++) {
         if (shot.indexOf(terminators[t]) >= 0) return terminators[t];
      }
      return false;
   }

   sequences.findServes = findServes;
   function findServes(shots) {
      if (!shots) return [];
      var serves = [];
      var rally = [];
      var serve_codes = '0456g'.split('');
      for (var s=0; s < shots.length; s++) {
         if (shots[s].length && serve_codes.indexOf(shots[s][0]) >= 0) {
            serves.push(shots[s]);
         }
      }
      return serves;
   }

   sequences.shotSplitter = shotSplitter;
   function shotSplitter(point) {
      var strokes = '0456fbrsvzopuylmhijktq';
      var stroke_array = strokes.split('');
      var shots = [];

      // remove any leading characters that are not considered strokes
      var leading_characters = true;
      while(leading_characters) { 
         if (point && '+-='.indexOf(point[0]) >= 0) {
            point = point.slice(1); 
         } else {
            leading_characters = false;
         }
      }

      var fodder = point.slice();
      var nextfodder;
      while(fodder.length) {
         for (var l=1; l < fodder.length; l++) {
            if (stroke_array.indexOf(fodder[l]) >= 0) { 
               shots.push(fodder.slice(0,l)); 
               nextfodder = fodder.slice(l); 
               break;
            }
         }
         if (l == fodder.length) {
            shots.push(fodder.slice(0,l)); 
            nextfodder = fodder.slice(l); 
         }
         fodder = nextfodder;
      }
      return shots;
   }

   var deuce_court_points = ['0-15', '15-0', '15-30', '30-15', '30-40', '40-30', '15-G', 'G-15'];
   var ad_court_points = ['15-15', '0-30', '30-30', '40-40', '15-40', '40-15', 'A-40', '40-A'];

   // shot pattern is an array of shots
   sequences.findShotPattern = findShotPattern;
   function findShotPattern(points, shot_pattern, reverse) {
      var matched_points = [];
      for (var p=0; p < points.length; p++) {
         var point = points[p];
         var point_shots = [];

         // take only the last shot in case mulptiple were coded
         if (point.serves) point_shots.push(point.serves[point.serves.length - 1]);
         // if (point.rally && point.rally.length) point.rally.forEach(e => point_shots.push(e));
         if (point.rally && point.rally.length) point.rally.forEach(function(e) { point_shots.push(e) });

         // saearch pattern is greater than # of shots in point
         if (shot_pattern.length > point_shots.length) continue;

         var fail = false;
         for (var s=0; s < shot_pattern.length; s++) {
            var shot_index = reverse ? point_shots.length - 1 - s : s;
            var pattern_index = reverse ? shot_pattern.length - 1 - s : s;
            var shot = point_shots[shot_index];
            var pattern = shot_pattern[pattern_index];
            if (!equalShots(shot, pattern)) fail = true;
         }
         if (!fail) matched_points.push(point);
      }
      return matched_points;

      function equalShots(shot, pattern) {
         var equal = true;
         if (shotType(pattern) && shotType(shot) != shotType(pattern)) equal = false;
         if (containsTerminator(pattern) && containsTerminator(shot) != containsTerminator(pattern)) equal = false;
         if (shotDirection(pattern) && shotDirection(shot) != shotDirection(pattern)) equal = false;
         if (shotDepth(pattern) && shotDepth(shot) != shotDepth(pattern)) equal = false;
         if (shotPosition(pattern) && shotPosition(shot) != shotPosition(pattern)) equal = false;
         if (shotFault(pattern) && shotFault(shot) != shotFault(pattern)) equal = false;
         return equal;
      }
   }

   if (typeof define === "function" && define.amd) define(sequences); else if (typeof module === "object" && module.exports) module.exports = sequences;
   this.sequences = sequences;
 
}();
