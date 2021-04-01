import pkg_resources, subprocess, unittest

class TestConsoleScripts(unittest.TestCase):
  def testConsoleScripts(self):
    """
    Make sure all console scripts defined in setup.py actually exist
    and that their imports work
    """
    for script in pkg_resources.iter_entry_points("console_scripts"):
      if script.dist.key == "astropath":
        with self.subTest(script=script.name):
          try:
            subprocess.run([script.name, "--help"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
          except subprocess.CalledProcessError as e:
            raise RuntimeError(e.stdout.decode())
