import os
import logging
from run_keras_server import app as application

if __name__ == "wsgi":	
	port = int(os.environ.get('PORT', 5000))
	print(port)
	application.run(host='0.0.0.0', port=port)