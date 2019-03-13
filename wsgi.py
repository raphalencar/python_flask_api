import os
from run_keras_server import app as application 

host = '0.0.0.0'
port = int(os.environ.get('PORT', 5000))
application.run(host=host, port=port)