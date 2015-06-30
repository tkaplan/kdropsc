tar -czf deploy.tar.gz .
ssh $DEVNODE bash -c "'rm -rf run/*'"
sftp -b putfile $DEVNODE
ssh $DEVNODE bash -c "'
cd run
tar -zxvf dep*
rm *.tar.gz
mkdir -p /home/dev/run/src/main/resources/lib
ln -s /usr/lib/libOpenCL.so.1 /home/dev/run/src/main/resources/lib/libOpenCL.so
mvn package
'"