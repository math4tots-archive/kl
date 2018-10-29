import os.net

// https://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf
void main() {
  runServer()
}

void runServer() {
  final socket = net.openSocket(net.PF_INET, net.SOCK_STREAM, 0)
  socket.bindIp4(net.htonl(net.INADDR_ANY), net.htons(8080))
  socket.listen(10)
  final outaddr = []
  final outsocket = socket.acceptIp4(outaddr)
  print(socket)
  print(outsocket)
  print('outaddr = ' + str(outaddr))
  final buf = Buffer(10)
  buf.set1(0, 10)
  buf.set1(1, 11)
  buf.set1(2, 12)
  outsocket.sendBuffer(buf, 0)
  socket.close()
}
