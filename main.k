import os.net

// https://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf
void main() {
  final socket = net.tryOpenSocket(net.PF_INET, net.SOCK_STREAM, 0).orDie()
  print(socket)
  socket.close()
}
