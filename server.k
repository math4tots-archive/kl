import os.net

// https://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf
void main() {
  runServer()
}

void runServer() {
  final socket = net.tryOpenSocket(net.PF_INET, net.SOCK_STREAM, 0).orDie()
  socket.bindIp4(net.htonl(net.INADDR_ANY), net.htons(8080))
  socket.listen(10)
  final outaddr = []
  final outsocket = socket.acceptIp4(outaddr)
  print(socket)
  print(outsocket)
  print('outaddr = ' + str(outaddr))
  socket.close()
}
