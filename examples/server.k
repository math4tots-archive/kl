import os.net

// https://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf
void main() {
  runServer()
}

void runServer() {
  int counter = 0
  final socket = net.openSocket(net.PF_INET, net.SOCK_STREAM, 0)
  socket.bindIp4(net.htonl(net.INADDR_ANY), net.htons(8080))
  socket.listen(10)
  print('Listening...')
  while (true) {
    final outaddr = []
    final outsocket = socket.acceptIp4(outaddr)
    print(socket)
    print(outsocket)
    print('outaddr = ' + str(outaddr))
    counter++
    outsocket.sendString('Hello visitor ' + str(counter))
    outsocket.close()
    print('sent message~')
  }
  socket.close()
}
