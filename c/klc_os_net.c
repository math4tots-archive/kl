#include "klc_os_net.h"

#if KLC_POSIX
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

typedef struct KLCNBufferView KLCNBufferView;

struct KLCNosZBnetZBSocket {
  KLC_header header;
  int domain;
  int type;
  int protocol;
  int socket;
};

static KLCNosZBnetZBSocket* mksocket(int socket, int domain, int type, int protocol) {
  KLCNosZBnetZBSocket* sock = (KLCNosZBnetZBSocket*) malloc(sizeof(KLCNosZBnetZBSocket));
  KLC_init_header(&sock->header, &KLC_typeosZBnetZBSocket);
  sock->domain = domain;
  sock->type = type;
  sock->protocol = protocol;
  sock->socket = socket;
  return sock;
}

KLCNTry* KLCNosZBnetZBtryOpenSocket(KLC_int domain, KLC_int type, KLC_int protocol) {
  #if KLC_POSIX
    KLCNTry* t;
    KLCNosZBnetZBSocket* sock;
    int s = socket((int) domain, (int) type, (int) protocol);
    if (s == -1) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    sock = mksocket(s, domain, type, protocol);
    t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) sock));
    KLC_release((KLC_header*) sock);
    return t;
  #else
    return NULL;
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryShutdown(KLCNosZBnetZBSocket* sock) {
  #if KLC_POSIX
    if (shutdown(sock->socket, SHUT_RDWR) != 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    } else {
      return KLCNTryZEnew(1, KLC_int_to_var(0));
    }
  #else
    return NULL;
  #endif
}

KLC_int KLCNosZBnetZBSocketZFGETZAfd(KLCNosZBnetZBSocket* sock) {
  #if KLC_POSIX
    return (KLC_int) sock->socket;
  #else
    return -1;
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryBindIp4(KLCNosZBnetZBSocket* sock, KLC_int addr, KLC_int port) {
  #if KLC_POSIX
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(struct sockaddr_in));
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = addr;
    sa.sin_port = port;
    if (bind(sock->socket, (struct sockaddr*) &sa, sizeof sa) == -1) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    return KLCNTryZEnew(1, KLC_int_to_var(0));
  #else
    KLC_errorf("TODO");
    return NULL;
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryClose(KLCNosZBnetZBSocket* sock) {
  #if KLC_POSIX
    if (close(sock->socket) != 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    return KLCNTryZEnew(1, KLC_int_to_var(0));
  #else
    KLC_errorf("TODO");
    return NULL;
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryListen(KLCNosZBnetZBSocket* sock, KLC_int backlog) {
  #if KLC_POSIX
    if (listen(sock->socket, (int) backlog) != 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    return KLCNTryZEnew(1, KLC_int_to_var(0));
  #else
    return KLC_failm("Socket.listen not supported for platform");
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryConnectIp4(KLCNosZBnetZBSocket* sock, KLC_int addr, KLC_int port) {
  #if KLC_POSIX
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(struct sockaddr_in));
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = addr;
    sa.sin_port = port;
    if (connect(sock->socket, (struct sockaddr*) &sa, sizeof sa) != 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    return KLCNTryZEnew(1, KLC_int_to_var(0));
  #else
    return KLC_failm("Socket.connectIp4 not support for platform");
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryAcceptIp4(KLCNosZBnetZBSocket* sock, KLCNList* outlist) {
  #if KLC_POSIX
    struct sockaddr_in sa;
    socklen_t alen = sizeof(struct sockaddr_in);
    KLCNList* list;
    KLCNTry* t;
    int s;
    KLCNosZBnetZBSocket* outsocket;
    memset(&sa, 0, sizeof(struct sockaddr_in));
    s = accept(sock->socket, (struct sockaddr*) &sa, &alen);
    if (s == -1) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    if (outlist != NULL) {
      KLCNListZFpush(outlist, KLC_int_to_var((KLC_int) sa.sin_addr.s_addr));
      KLCNListZFpush(outlist, KLC_int_to_var((KLC_int) sa.sin_port));
    }
    outsocket = mksocket(s, sock->domain, sock->type, sock->protocol);
    t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) outsocket));
    KLC_release((KLC_header*) outsocket);
    return t;
  #else
    return KLC_failm("Socket.acceptIp4 not supported for platform");
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtrySendBufferViewWithFlags(KLCNosZBnetZBSocket* sock, KLCNBufferView* b, KLC_int flags) {
  #if KLC_POSIX
    KLCNBuffer* bf = KLCNBufferViewZFGETbuffer(b);
    KLC_int bstart = KLCNBufferViewZFGETstart(b);
    KLC_int blen = KLCNBufferViewZFGETsize(b);
    char* cbuf = bf->buf + bstart;
    ssize_t s = send(sock->socket, cbuf, blen, (int) flags);
    KLC_release((KLC_header*) bf);
    if (s < 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    return KLCNTryZEnew(1, KLC_int_to_var((KLC_int) s));
  #else
    return KLC_failm("Socket.sendBuffer not supported for platform");
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryRecvBufferViewWithFlags(KLCNosZBnetZBSocket* sock, KLCNBufferView* b, KLC_int flags) {
  #if KLC_POSIX
    KLCNBuffer* bf = KLCNBufferViewZFGETbuffer(b);
    KLC_int bstart = KLCNBufferViewZFGETstart(b);
    KLC_int blen = KLCNBufferViewZFGETsize(b);
    char* cbuf = bf->buf + bstart;
    ssize_t s = recv(sock->socket, cbuf, blen, (int) flags);
    KLC_release((KLC_header*) bf);
    if (s < 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    return KLCNTryZEnew(1, KLC_int_to_var((KLC_int) s));
  #else
    return KLC_failm("Socket.recvBuffer not supported for platform");
  #endif
}

void KLC_deleteosZBnetZBSocket(KLC_header* robj, KLC_header** dq) {
  #if KLC_POSIX
    KLCNosZBnetZBSocket* s = (KLCNosZBnetZBSocket*) robj;

    /* TODO: Figure out what to do if shutdown fails */
    shutdown(s->socket, SHUT_RDWR);
  #endif
}

KLC_int KLCNosZBnetZBinetZAaddr(KLCNString* s) {
  #if KLC_POSIX
    return (KLC_int) inet_addr(s->utf8);
  #else
    KLC_errorf("TODO");
    return 0;
  #endif
}

KLC_int KLCNosZBnetZBhtonl(KLC_int x) {
  #if KLC_POSIX
    return (KLC_int) htonl((KLC_int) x);
  #else
    KLC_errorf("NOT SUPPORTED ON THIS PLATFORM");
    return 0; /* TODO */
  #endif
}

KLC_int KLCNosZBnetZBhtons(KLC_int x) {
  #if KLC_POSIX
    return (KLC_int) htons((KLC_int) x);
  #else
    KLC_errorf("NOT SUPPORTED ON THIS PLATFORM");
    return 0; /* TODO */
  #endif
}

KLC_int KLCNosZBnetZBntohs(KLC_int x) {
  #if KLC_POSIX
    return (KLC_int) ntohs((KLC_int) x);
  #else
    KLC_errorf("NOT SUPPORTED ON THIS PLATFORM");
    return 0; /* TODO */
  #endif
}

KLC_int KLCNosZBnetZBntohl(KLC_int x) {
  #if KLC_POSIX
    return (KLC_int) ntohl((KLC_int) x);
  #else
    KLC_errorf("NOT SUPPORTED ON THIS PLATFORM");
    return 0; /* TODO */
  #endif
}

KLC_int KLCNosZBnetZBINADDRZAANYZEinit() {
  #if KLC_POSIX
    return INADDR_ANY;
  #else
    KLC_errorf("NOT SUPPORTED ON THIS PLATFORM");
    return 0; /* TODO */
  #endif
}

KLC_int KLCNosZBnetZBAFZAINET6ZEinit() {
  #if KLC_POSIX
    return AF_INET6;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBAFZAINETZEinit() {
  #if KLC_POSIX
    return AF_INET;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBIPPROTOZATCPZEinit() {
  #if KLC_POSIX
    return IPPROTO_TCP;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBIPPROTOZAUDPZEinit() {
  #if KLC_POSIX
    return IPPROTO_UDP;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBSOCKZADGRAMZEinit() {
  #if KLC_POSIX
    return SOCK_DGRAM;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBSOCKZASEQPACKETZEinit() {
  #if KLC_POSIX
    return SOCK_SEQPACKET;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBSOCKZASTREAMZEinit() {
  #if KLC_POSIX
    return SOCK_STREAM;
  #else
    return -1;
  #endif
}
