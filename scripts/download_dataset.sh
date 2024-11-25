export SERVER_URL="https://dataverse.tdl.org"
export PERSISTENT_ID="doi:10.18738/T8/KENJXS"

curl -L -O -J -H "X-Dataverse-key" $SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID --output-dir ../bags