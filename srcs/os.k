#(append C_HEADERS 'klc_os.h')
#(append C_SOURCES 'klc_os.c')

extern String name
extern String sep

Try tryChdir(String path)
Try tryMkdir(String path)
Try tryListdir(String path)
Try tryGetcwd()

bool isfile(String path)
bool isdir(String path)

List listdir(String path) {
  return tryListdir(path).getOrElse(def var[String path](var error) {
    panic('Failed to listdir (' + path + '): ' + str(error))
    return null
  }).(List)
}

String getcwd() {
  return tryGetcwd().getOrElse(def var(var error) {
    panic('Failed getcwd: ' + str(error))
    return null
  }).(String)
}

String joinpath(List parts) {
  return sep.join(parts)
}

void chdir(String path) {
  tryChdir(path).getOrElse(def var[String path](var error) {
    panic('Failed to chdir (' + path + '): ' + str(error))
    return null
  })
}

void mkdir(String path) {
  tryMkdir(path).getOrElse(def var[String path](var error) {
    panic('Failed to mkdir (' + path + '): ' + str(error))
    return null
  })
}
