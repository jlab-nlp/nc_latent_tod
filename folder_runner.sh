#!/bin/bash
# run the script at argument 1 for each file in a directory (argument 2), s
SCRIPT_PATH="$1"
RUN_DIR="$2"
# if the user gave a third argument for pattern, use it:
PATTERN="*"
if [[ ! -z "$3" ]]; then
  PATTERN="$3"
fi

# if the script path doesn't start with src/nc_latent_tod, prepend it:
if [[ ! $SCRIPT_PATH == src/nc_latent_tod* ]]; then
  SCRIPT_PATH="src/nc_latent_tod/$SCRIPT_PATH"
fi
echo "running ${SCRIPT_PATH} with all files in ${RUN_DIR} :"
for f in $(find $RUN_DIR -name "${PATTERN}.json")
do
  echo "- $f"
done

for f in $(find $RUN_DIR -name "${PATTERN}.json")
do
  python "${SCRIPT_PATH}" "$f"
done