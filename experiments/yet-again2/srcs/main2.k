from c import size_t
from c import printf

class String {
  char* buffer
  size_t size

  delete {
    printf("String:delete called\n")
  }
}

extern int main(int argc, char** argv) {
  String s = $(String)
  0
}
