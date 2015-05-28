tar -czf deploy.tar.gz .
ssh $DEVNODE bash -c "'rm -rf *'"
sftp -b putfile $DEVNODE
ssh $DEVNODE bash -c "'
tar -zxvf dep*
rm *.tar.gz
mvn package
'"