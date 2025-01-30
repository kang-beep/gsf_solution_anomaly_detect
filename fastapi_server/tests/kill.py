import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('0.0.0.0', 8899))
except socket.error as e:
    print(f"Port 8899 is in use. Error: {e}")