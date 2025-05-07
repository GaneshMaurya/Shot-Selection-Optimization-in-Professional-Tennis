 
// Import the MCP parser
const mcp = require('./mcpParse')();
const fs = require('fs');
const path = require('path');

// Create output directories if they don't exist
const outputDir = path.join(__dirname, 'analysis_results');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
}

// Files to process - add your CSV filenames here (without the .csv extension)
const fileNames = ['charting-m-points-2010s','charting-m-points-2020s','charting-m-points-to-2009','charting-w-points-2010s','charting-w-points-2020s','charting-w-points-to-2009']; // Update with your actual files

// Track all matches for combined analysis
let allMatches = [];
let processedFiles = 0;

/**
 * Main function to process each file and then analyze the combined dataset
 */
function processFiles() {
    console.log(`Starting to process ${fileNames.length} files...`);
    loadNextFile(0);
}

/**
 * Recursive function to load files one by one
 */
function loadNextFile(index) {
    if (index >= fileNames.length) {
        console.log(`All files processed. Analyzing ${allMatches.length} total matches...`);
        analyzeAllData(allMatches);
        return;
    }
    
    const fileName = fileNames[index];
    console.log(`Loading file ${index + 1}/${fileNames.length}: ${fileName}.csv`);
    
    mcp.parseArchive(fileName, function(result) {
        if (result && result.matches && result.matches.length > 0) {
            console.log(`Successfully parsed ${result.matches.length} matches from ${fileName}.csv`);
            allMatches = allMatches.concat(result.matches);
            processedFiles++;
        } else {
            console.error(`Failed to parse matches from ${fileName}.csv`);
            if (result && result.errors) {
                console.error(`Errors: ${result.errors.length}`);
            }
        }
        
        // Process next file
        loadNextFile(index + 1);
    });
}

/**
 * Analyze the complete dataset after all files are processed
 */
function analyzeAllData(matches) {
    console.log(`Analyzing dataset with ${matches.length} matches...`);
    
    // Run each analysis function
    const basicStats = getBasicStats(matches);
    const shotStats = getShotDistribution(matches);
    const outcomeStats = getOutcomeAnalysis(matches);
    const patternStats = getPatternAnalysis(matches);
    
    // Write results to JSON files
    fs.writeFileSync(path.join(outputDir, 'basic_stats.json'), JSON.stringify(basicStats, null, 2));
    fs.writeFileSync(path.join(outputDir, 'shot_stats.json'), JSON.stringify(shotStats, null, 2));
    fs.writeFileSync(path.join(outputDir, 'outcome_stats.json'), JSON.stringify(outcomeStats, null, 2));
    fs.writeFileSync(path.join(outputDir, 'pattern_stats.json'), JSON.stringify(patternStats, null, 2));
    
    console.log('Analysis complete! Results saved to analysis_results directory.');
}

/**
 * Calculate basic dataset statistics
 */
function getBasicStats(matches) {
    let totalPoints = 0;
    const players = new Set();
    const tournamentTypes = {};
    const pointsPerMatch = [];
    const setsPerMatch = [];
    const matchYears = {};
    const surfaces = {};
    
    matches.forEach(match => {
        // Extract points
        const points = match.match.points();
        totalPoints += points.length;
        pointsPerMatch.push(points.length);
        
        // Add players
        const matchPlayers = match.match.players();
        matchPlayers.forEach(player => players.add(player));
        
        // Track tournament info
        if (match.tournament) {
            // Tournament name
            const tournamentName = match.tournament.name || 'Unknown';
            tournamentTypes[tournamentName] = (tournamentTypes[tournamentName] || 0) + 1;
            
            // Extract year from match date
            if (match.tournament.date) {
                const year = match.tournament.date.getFullYear();
                matchYears[year] = (matchYears[year] || 0) + 1;
            }
            
            // Track division (men's/women's)
            if (match.tournament.division) {
                surfaces[match.tournament.division] = (surfaces[match.tournament.division] || 0) + 1;
            }
        }
        
        // Count sets
        const setCount = match.match.sets ? match.match.sets().length : 0;
        setsPerMatch.push(setCount);
    });
    
    return {
        totalMatches: matches.length,
        totalPlayers: players.size,
        playerList: Array.from(players),
        totalPoints: totalPoints,
        averagePointsPerMatch: totalPoints / matches.length,
        pointsPerMatch: {
            min: Math.min(...pointsPerMatch),
            max: Math.max(...pointsPerMatch),
            average: pointsPerMatch.reduce((a, b) => a + b, 0) / pointsPerMatch.length,
            distribution: countOccurrences(pointsPerMatch)
        },
        setsPerMatch: {
            min: Math.min(...setsPerMatch),
            max: Math.max(...setsPerMatch),
            average: setsPerMatch.reduce((a, b) => a + b, 0) / setsPerMatch.length,
            distribution: countOccurrences(setsPerMatch)
        },
        tournamentTypes: tournamentTypes,
        matchesByYear: matchYears,
        matchesByDivision: surfaces,
        filesProcessed: processedFiles
    };
}

/**
 * Analyze shot distributions throughout dataset
 */
function getShotDistribution(matches) {
    const shotTypes = {};
    const serveTypes = {first: {}, second: {}};
    const rallyLengths = [];
    const shotDirections = {};
    const shotDepths = {};
    
    // Iterate through all matches and points
    matches.forEach(match => {
        const points = match.match.points();
        
        points.forEach(point => {
            // Process rally length
            const rallyLength = point.rally ? point.rally.length : 0;
            rallyLengths.push(rallyLength);
            
            // Process serve types - properly check the structure based on parser output
            if (point.serves && point.serves.length > 0) {
                const serve = point.serves[0];
                // First serve
                if (serve.charAt(0) === '4') serveTypes.first['wide'] = (serveTypes.first['wide'] || 0) + 1;
                else if (serve.charAt(0) === '5') serveTypes.first['body'] = (serveTypes.first['body'] || 0) + 1;
                else if (serve.charAt(0) === '6') serveTypes.first['T'] = (serveTypes.first['T'] || 0) + 1;
                else serveTypes.first['other'] = (serveTypes.first['other'] || 0) + 1;
            }
            
            // Second serve - check if first_serve property exists (indicates this was a second serve point)
            if (point.first_serve && point.first_serve.serves && point.first_serve.serves.length > 0) {
                const firstServe = point.first_serve.serves[0];
                if (firstServe.charAt(0) === '4') serveTypes.first['wide'] = (serveTypes.first['wide'] || 0) + 1;
                else if (firstServe.charAt(0) === '5') serveTypes.first['body'] = (serveTypes.first['body'] || 0) + 1;
                else if (firstServe.charAt(0) === '6') serveTypes.first['T'] = (serveTypes.first['T'] || 0) + 1;
                else serveTypes.first['other'] = (serveTypes.first['other'] || 0) + 1;
                
                // The point.serves would contain the second serve in this case
                if (point.serves && point.serves.length > 0) {
                    const secondServe = point.serves[0];
                    if (secondServe.charAt(0) === '4') serveTypes.second['wide'] = (serveTypes.second['wide'] || 0) + 1;
                    else if (secondServe.charAt(0) === '5') serveTypes.second['body'] = (serveTypes.second['body'] || 0) + 1;
                    else if (secondServe.charAt(0) === '6') serveTypes.second['T'] = (serveTypes.second['T'] || 0) + 1;
                    else serveTypes.second['other'] = (serveTypes.second['other'] || 0) + 1;
                }
            }
            
            // Process shot types in rallies
            if (point.rally && point.rally.length > 0) {
                point.rally.forEach(shot => {
                    // Shot type
                    let shotType = 'unknown';
                    if (shot.charAt(0) === 'f') shotType = 'forehand';
                    else if (shot.charAt(0) === 'b') shotType = 'backhand';
                    else if (shot.charAt(0) === 's') shotType = 'backhand_slice';
                    else if (shot.charAt(0) === 'r') shotType = 'forehand_slice';
                    else if (shot.charAt(0) === 'v') shotType = 'forehand_volley';
                    else if (shot.charAt(0) === 'z') shotType = 'backhand_volley';
                    else if (shot.charAt(0) === 'o') shotType = 'overhead';
                    else if (shot.charAt(0) === 'l') shotType = 'forehand_lob';
                    else if (shot.charAt(0) === 'm') shotType = 'backhand_lob';
                    
                    shotTypes[shotType] = (shotTypes[shotType] || 0) + 1;
                    
                    // Shot direction (if present)
                    for (let i = 0; i < shot.length; i++) {
                        if ('123'.includes(shot.charAt(i))) {
                            const direction = shot.charAt(i);
                            shotDirections[direction] = (shotDirections[direction] || 0) + 1;
                            break;
                        }
                    }
                    
                    // Shot depth (if present)
                    for (let i = 0; i < shot.length; i++) {
                        if ('789'.includes(shot.charAt(i))) {
                            const depth = shot.charAt(i);
                            shotDepths[depth] = (shotDepths[depth] || 0) + 1;
                            break;
                        }
                    }
                });
            }
        });
    });
    
    return {
        shotTypes: shotTypes,
        serveTypes: serveTypes,
        rallyLengths: {
            average: rallyLengths.reduce((a, b) => a + b, 0) / rallyLengths.length,
            distribution: countOccurrences(rallyLengths)
        },
        shotDirections: {
            '1': shotDirections['1'] || 0, // to the right
            '2': shotDirections['2'] || 0, // to the middle
            '3': shotDirections['3'] || 0  // to the left
        },
        shotDepths: {
            '7': shotDepths['7'] || 0, // shallow
            '8': shotDepths['8'] || 0, // deep
            '9': shotDepths['9'] || 0  // very deep
        }
    };
}

/**
 * Analyze point outcomes
 */
function getOutcomeAnalysis(matches) {
    const outcomes = {};
    const outcomeByShotType = {};
    const pointsEndedAtRallyLength = {};
    const errorTypes = {};
    
    matches.forEach(match => {
        const points = match.match.points();
        
        points.forEach(point => {
            // Point outcome
            const result = point.result || 'unknown';
            outcomes[result] = (outcomes[result] || 0) + 1;
            
            // Error type
            if (point.error) {
                errorTypes[point.error] = (errorTypes[point.error] || 0) + 1;
            }
            
            // Rally position where points end
            const rallyLength = point.rally ? point.rally.length : 0;
            pointsEndedAtRallyLength[rallyLength] = (pointsEndedAtRallyLength[rallyLength] || 0) + 1;
            
            // Last shot type and outcome
            if (point.rally && point.rally.length > 0) {
                const lastShot = point.rally[point.rally.length - 1];
                let shotType = 'unknown';
                
                if (lastShot.charAt(0) === 'f') shotType = 'forehand';
                else if (lastShot.charAt(0) === 'b') shotType = 'backhand';
                else if (lastShot.charAt(0) === 's') shotType = 'backhand_slice';
                else if (lastShot.charAt(0) === 'r') shotType = 'forehand_slice';
                else if (lastShot.charAt(0) === 'v') shotType = 'forehand_volley';
                else if (lastShot.charAt(0) === 'z') shotType = 'backhand_volley';
                else if (lastShot.charAt(0) === 'o') shotType = 'overhead';
                
                if (!outcomeByShotType[shotType]) outcomeByShotType[shotType] = {};
                outcomeByShotType[shotType][result] = (outcomeByShotType[shotType][result] || 0) + 1;
            }
        });
    });
    
    return {
        outcomes: outcomes,
        errorTypes: errorTypes,
        pointsEndedAtRallyLength: pointsEndedAtRallyLength,
        outcomeByShotType: outcomeByShotType
    };
}

/**
 * Analyze common patterns in the data
 */
function getPatternAnalysis(matches) {
    const servePlusOnePatterns = {};
    const commonRallySequences = {};
    const outcomeByServePlusOne = {};
    const serverWinRateByPattern = {};
    
    matches.forEach(match => {
        const points = match.match.points();
        
        points.forEach(point => {
            // Process serve+1 patterns (serve + first rally shot)
            if (point.serves && point.serves.length > 0 && point.rally && point.rally.length > 0) {
                const serve = point.serves[0].charAt(0);
                const firstReturnShot = point.rally[0].charAt(0);
                const servePlusOne = `${serve}-${firstReturnShot}`;
                
                // Count occurrences of this pattern
                servePlusOnePatterns[servePlusOne] = (servePlusOnePatterns[servePlusOne] || 0) + 1;
                
                // Track outcomes for this pattern
                if (!outcomeByServePlusOne[servePlusOne]) {
                    outcomeByServePlusOne[servePlusOne] = {};
                }
                const result = point.result || 'unknown';
                outcomeByServePlusOne[servePlusOne][result] = (outcomeByServePlusOne[servePlusOne][result] || 0) + 1;
                
                // Track server win rate by pattern
                if (!serverWinRateByPattern[servePlusOne]) {
                    serverWinRateByPattern[servePlusOne] = { total: 0, serverWins: 0 };
                }
                serverWinRateByPattern[servePlusOne].total++;
                if (point.winner === point.server) {
                    serverWinRateByPattern[servePlusOne].serverWins++;
                }
            }
            
            // Process 3-shot sequences in rallies
            if (point.rally && point.rally.length >= 3) {
                for (let i = 0; i < point.rally.length - 2; i++) {
                    const seq = `${point.rally[i].charAt(0)}-${point.rally[i+1].charAt(0)}-${point.rally[i+2].charAt(0)}`;
                    commonRallySequences[seq] = (commonRallySequences[seq] || 0) + 1;
                }
            }
        });
    });
    
    // Calculate win percentages for serve+1 patterns
    const patternWinRates = {};
    for (const [pattern, data] of Object.entries(serverWinRateByPattern)) {
        if (data.total >= 5) { // Only include patterns with sufficient sample size
            patternWinRates[pattern] = {
                total: data.total,
                winRate: (data.serverWins / data.total * 100).toFixed(2) + '%'
            };
        }
    }
    
    // Sort patterns by frequency to get top patterns
    const sortedServePlusOnePatterns = Object.entries(servePlusOnePatterns)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .reduce((obj, [key, value]) => {
            obj[key] = value;
            return obj;
        }, {});
    
    const sortedRallySequences = Object.entries(commonRallySequences)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .reduce((obj, [key, value]) => {
            obj[key] = value;
            return obj;
        }, {});
    
    return {
        topServePlusOnePatterns: sortedServePlusOnePatterns,
        topRallySequences: sortedRallySequences,
        outcomeByServePlusOne: outcomeByServePlusOne,
        patternWinRates: patternWinRates
    };
}

/**
 * Helper function to count occurrences in an array
 */
function countOccurrences(arr) {
    return arr.reduce((acc, curr) => {
        acc[curr] = (acc[curr] || 0) + 1;
        return acc;
    }, {});
}

// Start the analysis process
processFiles();