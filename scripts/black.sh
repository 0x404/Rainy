#! /bin/bash
CHECK="configs/ dataset/ model/ task/ utils/ test/"
CHECK="$CHECK runner.py launch.py"
black --check $CHECK 