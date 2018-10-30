import os.net

// https://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf
void main() {
  runClient()
}

void runClient() {
  for (int i = 0; i < 20000; i++) {
    print('about to create socket...')
    final socket = net.openSocket(net.PF_INET, net.SOCK_STREAM, 0)
    print('about to connect')
    socket.connectIp4(net.inet_addr("127.0.0.1"), net.htons(8080))
    print('connected')
    final msg = socket.recvAllBytesAsString()
    print('message = ' + msg)
    socket.close()
    print('closed socket')
  }
}
