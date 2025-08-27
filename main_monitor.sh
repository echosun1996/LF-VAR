#!/bin/bash
if [[ -n $1 ]]; then
  code=$1
else
  code=2
fi

echo "Start monitor remote repository"

# Record current latest commit hash
last_commit=$(git rev-parse HEAD)

# Enter monitoring loop
while true; do
    # Fetch remote updates without merging
    git fetch origin

    # Get current latest remote commit hash
    latest_commit=$(git rev-parse origin/master)

    # Compare local and remote commit hash
    if [[ "$last_commit" != "$latest_commit" ]]; then
        echo "Git repository has been updated. Executing main.sh..."

        # Update local code
        git pull origin master

        # Execute main.sh
        bash ./main.sh $code

        # Update last_commit to latest commit
        last_commit=$latest_commit
    fi

    sleep 2
done