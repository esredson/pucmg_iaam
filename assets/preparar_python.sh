apt-get update -y
apt-get install -y python
apt-get install -y pip
pip install pipreqs
pipreqs .
pip install -r requirements.txt