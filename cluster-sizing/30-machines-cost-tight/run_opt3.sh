#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <iterations> <output_directory>"
    echo "Example: $0 5 bo-classic"
    exit 1
fi

ITERATIONS=$1
OUTPUT_DIR=$2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the optimization iterations
for i in $(seq 1 $ITERATIONS); do
    echo "Running iteration $i/$ITERATIONS..."
    
    # Run the Python script
    python3 opt3.py
    
    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error: opt3.py failed on iteration $i"
        exit 1
    fi
    
    # Check if output files exist
    if [ ! -f "convergence.png" ] || [ ! -f "optimization_results.json" ]; then
        echo "Error: Expected output files not found on iteration $i"
        exit 1
    fi
    
    # Rename and move files
    mv convergence.png "$OUTPUT_DIR/convergence_$i.png"
    mv optimization_results.json "$OUTPUT_DIR/optimization_results_$i.json"
    
    echo "Iteration $i complete. Files moved to $OUTPUT_DIR/"
done

echo "All $ITERATIONS iterations completed successfully!"