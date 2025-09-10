#!/bin/bash

# go to dataset root
cd dataset/train

# resize in place (overwrite original)
find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.heic" \) \
  -exec mogrify -resize 640x640\> {} \;