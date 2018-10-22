import os.net

void main() {
  print(net.AF_INET)
  final socket = net.openSocket(net.AF_INET, net.SOCK_STREAM, net.IPPROTO_TCP)
  print(socket)
  socket.bindIp4(7654, net.htonl(net.INADDR_ANY))
  print('after bind')
  socket.shutdown()
}
