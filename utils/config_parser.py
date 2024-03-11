import configparser
config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}
config['roboflow'] = {}
config['roboflow']['key'] = ''  # add here your key
config['roboflow']['workspace'] = "research-g8szb"
config['roboflow']['project'] = "knotcounting"

with open('config.ini', 'w') as configfile:
  config.write(configfile)