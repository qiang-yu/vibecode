#!/bin/bash

# Target directory (defaults to current directory)
TARGET_DIR="."

echo "🔍 Scanning for .json files larger than 99MB in ${TARGET_DIR}..."
echo "----------------------------------------------------"

# File counter
count=0

# Find files matching *.json with size > 99M
while IFS= read -r -d '' file; do
    # Get initial size in MB
    size_mb=$(du -m "$file" | cut -f1)
    echo "📦 Found large file [${size_mb} MB]: $file"
    echo "   ⏳ Compressing with bzip2..."
    
    # Execute bzip2 compression (-v for verbose output)
    bzip2 -v "$file"
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Successfully compressed: ${file}.bz2"
        # Get compressed size in MB
        new_size_mb=$(du -m "${file}.bz2" | cut -f1)
        echo "   📉 Size reduced: ${size_mb} MB -> ${new_size_mb} MB"
        
        # Ensure original .json file is removed if it still exists
        if [ -f "$file" ]; then
            rm -f "$file"
            echo "   🗑️ Deleted original uncompressed file: $file"
        fi
    else
        echo "   ❌ Compression failed for: $file"
    fi
    echo "----------------------------------------------------"
    ((count++))
done < <(find "$TARGET_DIR" -type f -name "*.json" -size +99M -print0)

if [ $count -eq 0 ]; then
    echo "🎉 No .json files larger than 99MB were found."
else
    echo "✨ Finished processing! Compressed ${count} file(s) and cleaned up original .json files."
    echo "💡 Note: Remember to add the *.json.bz2 files to git before committing."
fi
