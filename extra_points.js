// extract_points.js
const fs = require('fs');
const mcp = require('./mcpParse')();

// Configure logging
const log_file = fs.createWriteStream('extraction_log.txt', {flags: 'w'});
function log(message) {
    console.log(message);
    log_file.write(message + '\n');
}

// File names to process (update these with your actual file names)
const fileNames = ['charting-m-points-2010s.csv','charting-m-points-2020s.csv','charting-m-points-to-2009.csv','charting-w-points-2010s.csv','charting-w-points-2020s.csv','charting-w-points-to-2009.csv'];
let allPoints = [];
let matchCount = 0;
let pointCount = 0;

log("Starting tennis point extraction...");
log(`Files to process: ${fileNames.join(', ')}`);

// Process each file sequentially
function processNextFile(index) {
    if (index >= fileNames.length) {
        // All files processed, save to CSV
        saveToCSV();
        return;
    }
    
    const fileName = fileNames[index];
    log(`\nProcessing file ${index + 1}/${fileNames.length}: ${fileName}.csv...`);
    
    mcp.parseArchive(fileName, function(result) {
        if (result && result.matches && result.matches.length > 0) {
            const currentMatches = result.matches.length;
            matchCount += currentMatches;
            log(`Successfully parsed ${currentMatches} matches from ${fileName}.csv`);
            
            // Extract points from each match
            result.matches.forEach((match, matchIndex) => {
                try {
                    // Access tournament info
                    const tournamentName = match.tournament ? match.tournament.name : 'unknown';
                    const players = match.match.players().join(' vs ');
                    
                    log(`Processing match ${matchIndex + 1}/${currentMatches}: ${players} (${tournamentName})`);
                    
                    // Extract points using the match accessor
                    const points = match.match.points();
                    log(`Found ${points.length} points in this match`);
                    
                    // Process each point
                    points.forEach(point => {
                        try {
                            // Flatten the point structure for CSV
                            const flatPoint = flattenPoint(point, match);
                            allPoints.push(flatPoint);
                            pointCount++;
                            
                            // Log every 1000 points for progress tracking
                            if (pointCount % 1000 === 0) {
                                log(`Processed ${pointCount} points so far...`);
                            }
                        } catch (pointError) {
                            log(`ERROR: Failed to process point: ${pointError.message}`);
                        }
                    });
                } catch (matchError) {
                    log(`ERROR: Failed to process match: ${matchError.message}`);
                }
            });
            
            // Process next file
            processNextFile(index + 1);
        } else {
            log(`ERROR: Failed to process ${fileName}.csv or no matches found`);
            processNextFile(index + 1);
        }
    });
}

// Flatten a point object to make it suitable for CSV storage
function flattenPoint(point, match) {
    // Match and tournament info
    const flat = {
        match_id: match.tournament ? match.tournament.name : 'unknown',
        player1: match.match.players()[0],
        player2: match.match.players()[1],
        tournament: match.tournament ? match.tournament.name : 'unknown',
        surface: match.tournament && match.tournament.surface ? match.tournament.surface : 'unknown',
        
        // Basic point info
        server: point.server !== undefined ? point.server : -1,
        score: point.score || '',
        winner: point.winner !== undefined ? point.winner : -1,
        outcome: point.result || 'Unknown',
    };
    
    // Serve information
    if (point.serves && point.serves.length > 0) {
        flat.serve_type = point.serves[0].charAt(0);
        flat.serve_direction = point.serves[0].match(/[123]/) ? point.serves[0].match(/[123]/)[0] : '0';
        flat.serve_depth = point.serves[0].match(/[789]/) ? point.serves[0].match(/[789]/)[0] : '0';
        flat.serve_full = point.serves[0];
    } else {
        flat.serve_type = '0';
        flat.serve_direction = '0';
        flat.serve_depth = '0';
        flat.serve_full = '';
    }
    
    // Second serve information
    flat.is_second_serve = point.first_serve ? 1 : 0;
    
    if (point.first_serve && point.first_serve.serves && point.first_serve.serves.length > 0) {
        flat.first_serve_type = point.first_serve.serves[0].charAt(0);
        flat.first_serve_full = point.first_serve.serves[0];
    } else {
        flat.first_serve_type = '';
        flat.first_serve_full = '';
    }
    
    // Rally information
    flat.rally_length = point.rally ? point.rally.length : 0;
    
    // Add rally shots (up to 5)
    if (point.rally) {
        const maxShots = Math.min(5, point.rally.length);
        for (let i = 0; i < maxShots; i++) {
            const shotName = `shot_${i+1}`;
            const shot = point.rally[i];
            
            flat[`${shotName}_type`] = shot.charAt(0);
            flat[`${shotName}_direction`] = shot.match(/[123]/) ? shot.match(/[123]/)[0] : '0';
            flat[`${shotName}_depth`] = shot.match(/[789]/) ? shot.match(/[789]/)[0] : '0';
            flat[`${shotName}_full`] = shot;
        }
    }
    
    // Add special information for the last shot
    if (point.rally && point.rally.length > 0) {
        const lastShot = point.rally[point.rally.length - 1];
        flat.last_shot_type = lastShot.charAt(0);
        flat.last_shot_direction = lastShot.match(/[123]/) ? lastShot.match(/[123]/)[0] : '0';
        flat.last_shot_depth = lastShot.match(/[789]/) ? lastShot.match(/[789]/)[0] : '0';
        flat.last_shot_full = lastShot;
    } else {
        flat.last_shot_type = '';
        flat.last_shot_direction = '0';
        flat.last_shot_depth = '0';
        flat.last_shot_full = '';
    }
    
    return flat;
}

// Save all points to CSV
function saveToCSV() {
    if (!allPoints.length) {
        log("No points to save!");
        return;
    }
    
    log(`\nPreparing to save ${allPoints.length} points to CSV...`);
    
    // Get all possible columns
    const columns = new Set();
    allPoints.forEach(point => {
        Object.keys(point).forEach(key => columns.add(key));
    });
    
    log(`Found ${columns.size} columns in the data`);
    
    // Create CSV header
    const header = Array.from(columns).join(',');
    
    // Create CSV rows
    const rows = allPoints.map(point => {
        return Array.from(columns)
            .map(col => {
                const value = point[col] !== undefined ? point[col] : '';
                // Escape commas and quotes
                if (typeof value === 'string') {
                    if (value.includes(',') || value.includes('"')) {
                        return `"${value.replace(/"/g, '""')}"`;
                    }
                }
                return value;
            })
            .join(',');
    });
    
    // Combine header and rows
    const csv = [header, ...rows].join('\n');
    
    // Save to file
    fs.writeFileSync('tennis_points.csv', csv);
    log(`Successfully saved ${allPoints.length} points to tennis_points.csv`);
    log(`Processed ${matchCount} matches total`);
    log("Point extraction complete!");
}

// Start processing
processNextFile(0);