// TODO
// Add Round to Tournament information
//
// finish combineMatchesPoints

module.exports = function() {

   // module container
   var mcp = {};

   // environment variables
   var mismatches = 0;

   // cached files
   var cache_default = './cache/';

   // external module dependencies
   var fs             = require('fs');
   var d3             = require('d3');
   var util           = require('util');
   var chardet        = require('chardet');
   var EventEmitter   = require('events').EventEmitter;
   var ProgressBar    = require('progress');

   var mo             = require('./matchObject');
   mcp.analysis       = require('./analysis');
   mcp.statistics     = require('./statistics');
   mcp.sequence       = require('./mcpSequence');

   mcp.mcpCSVparser = mcpCSVparser;
   function mcpCSVparser (file_name, cache_directory) {
      cache_directory = cache_directory ? cache_directory : cache_default;
      this.cache_directory = cache_directory;
      this.file_name = file_name
                       ? file_name.indexOf('.csv') > 0 ? file_name : file_name + '.csv'
                       : '';
      this.init();
   }

   util.inherits(mcpCSVparser, EventEmitter);

   mcpCSVparser.prototype.init = function () {
       var model;
       var self = this;
       self.on('loaded', function (points_array) {
          try {
             var result = self.loadMatches(self.file_name, points_array);
          }

          catch(err) {
             result = { error: 'Parsing Failed' };
          }

          if (result.error) {
             console.log('Error:', result.error);
             self.emit('complete', false);
          } else {
             self.emit('complete', result);
          }
       });
       if (!self.file_name) {
          console.log('Error: No File Name Given');
          self.emit('complete', false);
       } else {
          self.loadMatchArchive();
       }
   };

   mcpCSVparser.prototype.loadMatchArchive = function () {
     var self = this;
     var targetURL = self.cache_directory + self.file_name;
     console.log('Loading File:' + targetURL + '\r\nPlease be patient if file is large...');
     var chard = chardet.detectFileSync(targetURL);
     if (chard == 'ISO-8859-1' || chard == 'UTF-8') {
        var encoding = 'utf8';
     } else {
        var encoding = 'utf16le';
     }
     fs.readFile(targetURL, encoding, function(err, data) {
        if (err) {
           self.emit('error', err);
        } else {
           console.log('\r\nParsing CSV File...');
           var points_array = d3.csv.parse(data)
           self.emit('loaded', points_array);
        }
     });
   };

   // read .csv file and separate into matches
   mcpCSVparser.prototype.loadMatches = function (archiveURL, points_array) {
      console.log(points_array.length + ' points loaded\r\nSeparating Matches...');
      var match_points = [];
      var matches = [];
      var match_id;
      points_array.forEach(function(point) {
         if (point.match_id != match_id) {
            if (match_points.length) matches.push({ match_id: match_id, points: match_points });
            match_id = point.match_id;
            match_points = [point];
         } else {
            match_points.push(point);
         }
      });
      console.log(matches.length + ' matches');
      return matches;
   }

   mcp.parseArchive = function (file_name, callback) {
      var validate = (typeof callback == 'string' && callback == 'validate') ? true : false;
      var csv_parser = new mcpCSVparser(file_name);

      csv_parser.on('error', function (error) {
          console.log('Error parsing:', file_name, error);
        });

      csv_parser.on('complete', function (matches) {
         if (matches) {
            console.log('Matches Loaded\r\nParsing Shot Sequences...');
            parseMatches(matches);      
         }
      });

      function parseMatches(match_array) {
         var parsed_matches = [];
         var errors = [];
         var bar = new ProgressBar(':bar', { total: match_array.length });
         for (var m=0; m < match_array.length; m++) {
            if (validate) {
               // compares two methods of parsing
               var result = csv_parser.validateMatch(match_array[m]);
            } else {
               // utilizeds only shot sequence parsing
               var result = csv_parser.parseMatch(match_array[m]);
            }

            if (!result) {
               errors.push('ERROR: ' + match_array[m].match_id + ' => No Result');
            } else if (result.error) {
               errors.push('ERROR: ' + match_array[m].match_id + ' => ' + result);
            } else {
               parsed_matches.push(result);
            }
            bar.tick();
         }
         console.log(parsed_matches.length + ' Matches Successfully Parsed');
         if (errors.length) console.log(errors.length + ' Parsing Errors');
         mcp.matches = parsed_matches;
         mcp.errors = errors;
         if (typeof callback == 'function') callback({ matches: parsed_matches, errors: errors });
      }
   }

   mcpCSVparser.prototype.parseMatch = function (match) {
      if (!match || !match.points) return false;
      var points = match.points;

      // parse match_id for player and tournament data
      var players = parsePlayers(points[0].match_id);
      var tournament = parseTournament(points[0].match_id);
      tournament.date = parseDate(points[0].match_id);

      // create UMO to generate valid point progression
      var UMO_shot_seq = new mo.matchObject();
      // set player names
      UMO_shot_seq.players(players[0], players[1]);

      // if # of sets or final_set_tiebreak different than defaults
      if (tournament.sets) { UMO_shot_seq.options({match: {sets: tournament.sets}}); }
      if (tournament.final_set_tiebreak != undefined) {
         UMO_shot_seq.options({match: {final_set_tiebreak: tournament.final_set_tiebreak}});
      }

      for (var p=0; p < points.length; p++) {

         // use UMO to validate point
         var serves = getServes(points[p]);
         var parsed_point_sequence = mcp.sequence.pointParser(serves);

         // null point encountered
         if (parsed_point_sequence.continue || !parsed_point_sequence.winner) {
            continue;
         }

         var result_shot_seq = UMO_shot_seq.push(parsed_point_sequence);

         // abort with error
         if (result_shot_seq.error) { return result_shot_seq; }
      }

      // return players and score
      return { match: UMO_shot_seq, tournament: tournament };
   }

   function getServes(value) {
      return [value['1st'], value['2nd']];
   }

   function parsePlayers(match_id) {
      // not a very reobust parse at the moment!
      var players = match_id.split('_').join(' ').split('-').slice(4,6);
      return players;
   }

   function parseDate(match_id) {
      var splitMatchID = match_id.split('-');
      var dt = splitMatchID[0];
      var date = dt.slice(4,6) + '-' + dt.slice(6,8) + '-' + dt.slice(0,4);
      var match_date = new Date(date);
      return match_date;
   }

   function parseTournament(match_id) {
      var tournament = {};
      var splitMatchID = match_id.split('-');
      tournament.name = (splitMatchID.length > 2) ? normalizeTournament(splitMatchID[2]) : '';
      tournament.division = (splitMatchID.length > 1) ? 
                            ['M', 'W'].indexOf(splitMatchID[1]) >= 0 ? splitMatchID[1] : '' : '';

      if (tournament.name.indexOf('ITF ') == 0) {
         tournament.name = tournament.name.slice(4);
         tournament.tour = 'itf';
      }

      if (tournament.name.indexOf('WTA ') == 0) {
         tournament.tour = 'wta';
      }

      if (tournament.name == 'French Open') tournament.name = 'Roland Garros';
      if (tournament.name.indexOf('Davis Cup') == 0) tournament.name = 'Davis Cup';
      if (tournament.name.indexOf('Fed Cup') == 0) tournament.name = 'Fed Cup';

      if (tournament.name.match(' CH'+'$') == ' CH') {
         tournament.name = tournament.name.slice(0, tournament.name.length - 3);
         tournament.tour = 'ch';
      }

      if (tournament.name.indexOf('WTC ') == 0 || tournament.name.match(' WCT'+'$') == ' WCT') {
         tournament.tour = 'wct';
      }

      if (tournament.name.match(' Masters'+'$') == ' Masters') {
         tournament.tour = 'atp';
      }

      if (tournament.name.match(' Q'+'$') == ' Q') {
         tournament.name = tournament.name.slice(0, tournament.name.length - 2);
         tournament.draw = 'qual';
      }

      if (tournament.name.match(/\sF[0-9]+$/)) {
         var temp = tournament.name.split(' ');
         tournament.name = temp.slice(0, temp.length - 1).join(' ');;
         tournament.tour = 'fu';
      }

      if (tournament.name.match(/\s[0-9]+K/)) {
         var temp = tournament.name.split(' ');
         tournament.name = temp.slice(0, temp.length - 1).join(' ');;
         tournament.prize_money = '$' + temp[temp.length - 1].replace('K', ',000').trim();
         tournament.tour = 'itf';
      }

      var grand_slam_scoring = ['Australian Open', 'US Open', 'Roland Garros', 'Wimbledon', 'Davis Cup'];
      var no_final_set_tiebreak = ['Australian Open', 'Roland Garros', 'Wimbledon', 'Fed Cup', 'Olympics'];

      if (grand_slam_scoring.indexOf(tournament.name) >= 0) {
         if (tournament.division == 'W') {
            tournament.sets = 3;
         } else if (tournament.division == 'M') {
            tournament.sets = 5;
         }
      }

      if (no_final_set_tiebreak.indexOf(tournament.name) >= 0) {
         tournament.final_set_tiebreak = false;
      }

      return tournament;
   }

   function normalizeTournament(tournament) {
      var particles = ['de', 'di', 'du', 'van', 'von', 'ten'];
      var t_split = tournament.split('_');
      var normalized = t_split.map(function(e, i) {
         e = e.trim();
         if (!e) return;
         if (e === e.toUpperCase()) return e; // acronym
         if (i == 0 || i == t_split.length - 1 || particles.indexOf(e.toLowerCase()) < 0) {
            return e[0].toUpperCase() + e.slice(1); 
         } else {
            return e.toLowerCase();
         }
      }).join(' ');
      return normalized;
   }

   // used for debugging, compares two parsing methods
   mcpCSVparser.prototype.validateMatch = function (match) {
      if (!match || !match.points) return false;
      var points = match.points;

      var debug_list = [
         '20140118-W-Australian_Open-R32-Maria_Sharapova-Alize_Cornet',
         '19811130-W-Australian_Open-F-Martina_Navratilova-Chris_Evert',
         '20131005-M-Tokyo-SF-Nicolas_Almagro-Juan_Martin_Del_Potro',
         '19991122-M-Tour_Finals-RR-Pete_Sampras-Andre_Agassi'
      ];

      var debug_file = (debug_list.indexOf(points[0].match_id) >= 0) ? true : false;

      // parse match_id for player and tournament data
      var players = parsePlayers(points[0].match_id);
      var tournament = parseTournament(points[0].match_id);

      // create two separate UMOs
      var UMO_PtsAfter = new mo.matchObject();
      var UMO_shot_seq = new mo.matchObject();
      UMO_PtsAfter.players(players[0], players[1]);
      UMO_shot_seq.players(players[0], players[1]);

      // if # of sets or final_set_tiebreak different than defaults
      if (tournament.sets) {
         UMO_PtsAfter.options({match: {sets: tournament.sets}});
         UMO_shot_seq.options({match: {sets: tournament.sets}});
      }
      if (tournament.final_set_tiebreak != undefined) {
         UMO_PtsAfter.options({match: {final_set_tiebreak: tournament.final_set_tiebreak}});
         UMO_shot_seq.options({match: {final_set_tiebreak: tournament.final_set_tiebreak}});
      }

      // necessary for PtsAfter() transformation of GM to point score
      var last_point;

      for (var p=0; p < points.length; p++) {

         // use UMO to validate point
         var ppp = ptsAfter(points[p]);
         var serves = getServes(points[p]);
         var pps = pointParser(serves);

         // null point encountered
         if (ppp.continue || pps.continue) continue;

         if (!ppp.point || !pps.winner) {
            console.log('missing point', ppp.point, pps.winner);
            continue;
         }

         var result_PtsAfter = UMO_PtsAfter.push(ppp.point);
         var result_shot_seq = UMO_shot_seq.push({ winner: pps.winner, shots: pps.shots, rally: pps.rally });

         // compare points returned by UMO for each parser
         if (debug_file && result_PtsAfter.point && result_shot_seq.winner && result_PtsAfter.point.point != result_shot_seq.point.winner) {
            console.log(pps.winner, '\t' + result_shot_seq.point.winner, '\t' + ppp.point, '\t' + result_PtsAfter.point.point, '\t' + pps.code);
         }

         // abort with error
         if (result_shot_seq.error) return result_shot_seq;

         // set last_point, necessary for PtsAfter() transformation of GM to point score
         last_point = ppp.point;
      }

      // if the two UMOs have differeing scores, display them
      if (UMO_shot_seq.score().match_score.trim() != UMO_PtsAfter.score().match_score.trim()) {
         mismatches += 1;
         console.log('parsing mismatch #: ', mismatches, ' ', points[0].match_id);
         console.log(UMO_PtsAfter.score().match_score, ' / ', UMO_shot_seq.score().match_score);
      }

      // return players and score
      return { players: UMO_shot_seq.players(), score: UMO_shot_seq.score().match_score };

      // small parser for ptsAfter column of MCP .csv files
      function ptsAfter(value) {
         if (!value || !value.PtsAfter) return { continue: true };
         var point = value.PtsAfter.slice();

         // reverse point if second player serving
         if (points[p].Svr == 2) { point = point.split('-').reverse().join('-'); }
         point = point.replace('AD', 'A');

         if (last_point && point == 'GM') {
            var point = last_point.replace('A', 50);
            var score = point.split('-');
            var winner = parseInt(score[0]) > parseInt(score[1]) ? 0 : 1;
            score[winner] = 'G';
            point = score.join('-');
         }
         return { point: point };
      }
   }

   mcp.playerMatches = playerMatches;
   function playerMatches(match_array, player_name) {
      var matches = [];
      match_array.forEach(function(match) {
         var players = match.match.players();
         if (players[0].search(player_name) >= 0 || players[1].search(player_name) >= 0) {
            matches.push(match);
         }
      });
      return matches;
   }

   // facilitate aggregate stats across a number of matches
   // searched for player is player 0 in combined results
   mcp.combineMatchesPoints = combineMatchesPoints;
   function combineMatchesPoints (player_name, matches) {
      var c_points = [];
      for (var m=0; m < matches.length; m++) {
         var m_players = matches[m].match.players();
         var m_points = matches[m].match.points();
         if (m_players[0].search(player_name) >= 0 && m_players[1].search(player_name) >= 0) continue;
         if (m_players[0].search(player_name) >= 0) {
            Array.prototype.push.apply(c_points, m_points);
            continue;
         } else if (m_players[1].search(player_name) >= 0) {
            var player_index = 1;
         } else {
            continue;
         }
         m_points.forEach(function(point) {
            var new_point = (JSON.parse(JSON.stringify(point)));
            new_point.server = 1 - new_point.server;
            new_point.winner = 1 - new_point.winner;
            new_point.match = m;
            c_points.push(new_point);
         });
      }
      return c_points;
   }

   // shot pattern is an array of shots
   mcp.findShotPattern = findShotPattern;
   function findShotPattern(points, shot_pattern, reverse) {
      var matched_points = [];
      for (var p=0; p < points.length; p++) {
         var point = points[p];
         var point_shots = [];

         // take only the last shot in case mulptiple were coded
         if (point.serves) point_shots.push(point.serves[point.serves.length - 1]);
         if (point.rally && point.rally.length) point.rally.forEach(e => point_shots.push(e));

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

   mcp.localCacheList = localCacheList;
   function localCacheList() {
      var ignore_list = [];
      ignore_list.push('20151104-M-20151103-R64-Viktor_Troicki-Jack_Sock.html');                // duplicate file with incorrect tournament name
      ignore_list.push('1990409-W-Amelia_Island-F-Steffi_Graf-Arantxa_Sanchez_Vicario.html');   // incorrect date (file duplicated)
      ignore_list.push('20151022-M-Vienna-R16-Jo_Wilfried_Tsonga-Lukas_Rosol.html');            // incomplete file
      var files = fs.readdirSync(cache_default);
      files = files.filter(function(f) { return f.indexOf('DS_Store') < 0 && f.indexOf('un~') < 0; });
      files = files.filter(function(f) { return ignore_list.indexOf(f) < 0 });
      return files;
   }

   return mcp;
}
