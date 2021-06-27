hash=`find ./archive -type f -exec md5sum {} \; | md5sum | awk '{ print $1 }'`
ts=`date "+%y%m%d%H%M%S"`
fname="submission_${ts}_${hash:0:8}.zip"

cd ./archive
zip -qr ../submit/$fname *

echo "---> submit/$fname"