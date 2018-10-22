// Make sure that 'os' is imported.
// This way KLC_os.{c,h} are included
import os

extern String sep

String join(List parts) {
  return sep.join(parts)
}
