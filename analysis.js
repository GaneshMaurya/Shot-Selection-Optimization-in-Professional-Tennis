!function() { 

   // module container
   var analyze = {};

   // Analysis

   analyze.serveAnalysis = serveAnalysis;
   function serveAnalysis(points, verbose) {
      var serve_types = {};
      var invalid_serves = 0;
      var multiple_serves = 0;
      for (var p=0; p < points.length; p++) {
         var point = points[p];
         if (point.serves.length > 1) multiple_serves += 1;
         if (point.first_serve && point.first_serve.serves && point.first_serve.serves.length > 1) multiple_serves += 1;
         if (point.serves.length == 0) invalid_serves += 1;
         if (point.first_serve && point.first_serve.serves && point.first_serve.serves.length == 0) invalid_serves += 1;
         if (point.serves.length == 1) {
            if (!serve_types[point.serves[0]]) {
               serve_types[point.serves[0]] = 1;
            } else {
               serve_types[point.serves[0]] += 1;
            }
         }
      }
      var analysis = {
         serve_types: serve_types,
         invalid_serves: invalid_serves,
         multiple_serves: multiple_serves
      };
      return analysis
   }

   analyze.matchesServeAnalysis = matchesServeAnalysis;
   function matchesServeAnalysis(matches, verbose) {
      var serve_types = {};
      var invalid_serves = 0;
      var multiple_serves = 0;
      matches.forEach(function(match, i) {
         var analysis = serveAnalysis(match.match.points(), verbose);
         invalid_serves += analysis.invalid_serves;
         multiple_serves += analysis.multiple_serves;
         var serve_keys = Object.keys(analysis.serve_types);
         serve_keys.forEach(function(s) {
            if (!serve_types[s]) {
               serve_types[s] = analysis.serve_types[s];
            } else {
               serve_types[s] += analysis.serve_types[s];
            }
         });
      });
      var matches_analysis = {
         serve_types: serve_types,
         invalid_serves: invalid_serves,
         multiple_serves: multiple_serves
      };
      return matches_analysis
   }

   analyze.rallyDepth = rallyDepth;
   function rallyDepth(points, verbose) {
      var no_depth_or_finish = [];
      var rally_count = 0;
      var return_count = 0;
      var return_depth = 0;
      var return_finish = 0;
      var depth_count = 0;
      for (var p=0; p < points.length; p++) {
         if (points[p].rally.length) {
            return_count += 1;
            var shots = points[p].rally;
            var ros = shots[0];
            var depth = findDepth(ros);
            var finish = findTerminator(ros);

            if (depth) {
               return_depth += 1;
            }
            if (finish) {
               return_finish += 1;
            }
            if (!depth && !finish) {
               no_depth_or_finish.push(ros);
            }

            if (shots.length > 1) {
               for (var r=1; r < shots.length; r++) {
                  rally_count += 1;
                  var rally_depth = findDepth(shots[r]);
                  if (rally_depth) {
                     depth_count += 1;
                     if (verbose) console.log(points[p].serves, shots);
                  }
               }
            }
         }
      }

      function findDepth(shot) {
         for (var i=0; i < shot.length; i++) {
            if ('789'.split('').indexOf(shot[i]) >= 0) return true;
         }
         return false;
      }

      function findTerminator(shot) {
         for (var i=0; i < shot.length; i++) {
            if ('*@#'.split('').indexOf(shot[i]) >= 0) return true;
         }
         return false;
      }

      var analysis = { 
         points: points.length, 
         returns: return_count, 
         return_finish: return_finish, 
         return_depth: return_depth, 
         return_other: no_depth_or_finish, 
         shots: rally_count, 
         rally_depth: depth_count
      }
      return analysis;
   }

   analyze.matchesWithRallyDepth = matchesWithRallyDepth;
   function matchesWithRallyDepth(matches, verbose) {
      var rally_shots = 0;
      var rally_depth = 0;
      var matches_with_rally_depth = [];
      matches.forEach(function(match, i) {
         var analysis = rallyDepth(match.match.points(), verbose);
         rally_shots += analysis.shots;
         if (analysis.rally_depth > 0) {
            matches_with_rally_depth.push(i);
            rally_depth += analysis.rally_depth;
         }
      });
      return { rally_shots: rally_shots, rally_depth: rally_depth, matches: matches_with_rally_depth };
   }

   if (typeof define === "function" && define.amd) define(analyze); else if (typeof module === "object" && module.exports) module.exports = analyze;
   this.analyze = analyze;
 
}();
