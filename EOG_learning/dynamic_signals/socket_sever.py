import ipaddress
import socket
import sys

my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = '127.0.0.1'
port = 8888
ip_address = (ip, port)
my_socket.bind(ip_address)

while True:
    data, _ = my_socket.recvfrom(32)
    print('Server received:', data.decode('utf-8'))