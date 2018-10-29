import os.net

// https://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf
void main() {
  runClient()
}

void runClient() {
  final socket = net.openSocket(net.PF_INET, net.SOCK_STREAM, 0)
  print(socket)
  socket.connectIp4(net.inet_addr("127.0.0.1"), net.htons(8080))
  final buf = Buffer(1024)
  final nread = socket.recvBuffer(buf, 0)
  print(buf.dumpstr())
  print('nread = ' + str(nread))
  socket.close()
}
