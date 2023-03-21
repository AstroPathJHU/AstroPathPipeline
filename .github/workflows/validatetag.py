import argparse, re, subprocess

def validatetag(*, tag, ok_if_version_unchanged):
  regex = r"v([0-9]+)\.([0-9]+)\.([0-9]+)$"
  match = re.match(regex, tag)
  if not match:
    raise ValueError(f"{tag} does not match {regex}")
  tag_version = int(match.group(1)), int(match.group(2)), int(match.group(3))

  git_describe_tags = subprocess.run(["git", "describe", "--tags"], check=True, capture_output=True)
  describe_output = git_describe_tags.stdout.strip().decode("ascii")
  previous_tag = describe_output.split("-")[0]
  describe_regex = regex.replace("$", "(-[0-9]+-g[0-9a-f]+)?")
  match = re.match(describe_regex, describe_output)
  previous_version = int(match.group(1)), int(match.group(2)), int(match.group(3))
  is_dev = match.group(4) is not None

  if tag_version == previous_version:
    if is_dev:
      if not ok_if_version_unchanged:
        raise ValueError(f"Did not increment the version number from {previous_tag}")
    else:
      pass #ok, that just means you're running on the same commit that got tagged
  elif tag_version < previous_version:
    raise ValueError(f"Version is decremented from {previous_tag} to {tag}")
  else:
    git_tag_l = subprocess.run(["git", "tag", "-l", tag], check=True, capture_output=True)
    if git_tag_l.stdout:
      raise ValueError(f"{tag} already exists")

def main(args=None):
  p = argparse.ArgumentParser()
  p.add_argument("tag")
  p.add_argument("--ok-if-version-unchanged", action="store_true")
  args = p.parse_args()
  validatetag(**args.__dict__)

if __name__ == "__main__":
  main()

