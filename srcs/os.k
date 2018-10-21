#(append C_HEADERS 'klc_os.h')
#(append C_SOURCES 'klc_os.c')

extern String name
extern String sep

bool chdirOrFalse(String path)
bool mkdirOrFalse(String path)
bool isfile(String path)
bool isdir(String path)
List listdirOrNull(String path)
String getcwdOrNull()

List listdir(String path) {
  final ret = listdirOrNull(path)
  if (ret is null) {
    panic('Failed to listdir ' + str(path))
  }
  return ret
}

String getcwd() {
  final ret = getcwdOrNull()
  if (ret is null) {
    panic('Failed to getcwd')
  }
  return ret
}

String joinpath(List parts) {
  return sep.join(parts)
}

void chdir(String path) {
  if (!chdirOrFalse(path)) {
    panic('chdir failed for ' + str(path))
  }
}

void mkdir(String path) {
  if (!mkdirOrFalse(path)) {
    panic('mkdir failed for ' + path)
  }
}
