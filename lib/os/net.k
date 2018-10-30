#(append C_HEADERS 'klc_os_net.h')
#(append C_SOURCES 'klc_os_net.c')

extern int INADDR_ANY
extern int AF_INET
extern int AF_INET6
extern int SOCK_STREAM
extern int SOCK_DGRAM
extern int SOCK_SEQPACKET
extern int IPPROTO_TCP
extern int IPPROTO_UDP

// https://stackoverflow.com/questions/6729366/what-is-the-difference-between-af-inet-and-pf-inet-in-socket-programming
int PF_INET = AF_INET
int PF_INET6 = AF_INET6

extern class Socket {
  Try tryShutdown()
  Try tryBindIp4(int addr, int port)
  Try tryClose()
  Try tryListen(int backlog)
  Try tryConnectIp4(int addr, int port)
  Try tryAcceptIp4(List addr)
  Try trySendBufferViewWithFlags(BufferView buffer, int flags)
  Try tryRecvBufferViewWithFlags(BufferView buffer, int flags)

  // file descriptor for sockets is an implementation detail.
  // Might not return a meaningful value
  int GET_fd()

  void shutdown() {
    this.tryShutdown().orDie()
  }

  void bindIp4(int addr, int port) {
    this.tryBindIp4(addr, port).orDie()
  }

  void close() {
    this.tryClose().orDie()
  }

  void listen(int backlog) {
    this.tryListen(backlog).orDie()
  }

  void connectIp4(int addr, int port) {
    this.tryConnectIp4(addr, port).orDie()
  }

  Socket acceptIp4(List addr) {
    return this.tryAcceptIp4(addr).orDie().(Socket)
  }

  Try trySendBufferWithFlags(Buffer buffer, int flags) {
    return this.trySendBufferView(buffer.view(0, buffer.size))
  }

  Try tryRecvBufferWithFlags(Buffer buffer, int flags) {
    return this.tryRecvBufferView(buffer.view(0, buffer.size))
  }

  Try trySendBuffer(Buffer buffer) {
    return this.trySendBufferWithFlags(buffer, 0)
  }

  Try tryRecvBuffer(Buffer buffer) {
    return this.tryRecvBufferWithFlags(buffer, 0)
  }

  Try trySendBufferView(BufferView buffer) {
    return this.trySendBufferViewWithFlags(buffer, 0)
  }

  Try tryRecvBufferView(BufferView buffer) {
    return this.tryRecvBufferViewWithFlags(buffer, 0)
  }

  int sendBufferViewWithFlags(BufferView buffer, int flags) {
    return this.trySendBufferViewWithFlags(buffer, flags).orDie().(int)
  }

  int recvBufferViewWithFlags(BufferView buffer, int flags) {
    return this.tryRecvBufferViewWithFlags(buffer, flags).orDie().(int)
  }

  int sendBufferView(BufferView buffer) {
    return this.trySendBufferView(buffer).orDie().(int)
  }

  int recvBufferView(BufferView buffer) {
    return this.tryRecvBufferView(buffer).orDie().(int)
  }

  int sendBufferWithFlags(Buffer buffer, int flags) {
    return this.trySendBufferWithFlags(buffer, flags).orDie().(int)
  }

  int recvBufferWithFlags(Buffer buffer, int flags) {
    return this.tryRecvBufferWithFlags(buffer, flags).orDie().(int)
  }

  int sendBuffer(Buffer buffer) {
    return this.trySendBuffer(buffer).orDie().(int)
  }

  int recvBuffer(Buffer buffer) {
    return this.tryRecvBuffer(buffer).orDie().(int)
  }

  String Repr() {
    return '<os.net.Socket (' + str(this._fd) + ')>'
  }
}

int inet_addr(String addr)
int htonl(int x)
int htons(int x)
int ntohl(int x)
int ntohs(int x)

Try tryOpenSocket(int domain, int type, int protocol)

Socket openSocket(int domain, int type, int protocol) {
  return tryOpenSocket(domain, type, protocol).orDie().(Socket)
}
