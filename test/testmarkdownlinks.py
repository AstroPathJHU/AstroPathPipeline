import bs4, functools, git, itertools, marko.ext.toc, pathlib, re, slugify, unittest

class GithubTocRendererMixin(marko.ext.toc.TocRendererMixin):
  def render_heading(self, element):
    children = self.render_children(element)
    slug = re.sub(r"<.+?>", "", children)
    slug = re.sub(r"(?<=[0-9])[.](?=[0-9])", r"", slug)
    slug = slug.replace("_", "UNDERSCOREUNDERSCORE")
    slug = slug.replace("#", "HASHTAGHASHTAG")
    slug = slugify.slugify(slug)
    slug = slug.replace("underscoreunderscore", "_")
    slug = slug.replace("hashtaghashtag", "")
    self.headings.append((int(element.level), slug, children))
    return '<h{0} id="{1}">{2}</h{0}>\n'.format(element.level, slug, children)

class GithubToc:
  def __init__(self):
    self.renderer_mixins = [GithubTocRendererMixin]

thisfolder = pathlib.Path(__file__).parent
mainfolder = thisfolder.parent

class LinkError(Exception): pass

@functools.lru_cache()
def linksandanchors(filename):
  try:
    with open(filename, encoding="utf-8") as f:
      markdown = f.read()
    html = marko.Markdown(extensions=[GithubToc()])(markdown)
    soup = bs4.BeautifulSoup(html, features="lxml")
    links = soup.findAll("a", attrs={"href": re.compile(".*")})
    anchors = sum((soup.findAll(f"h{i}", attrs={"id": re.compile(".*")}) for i in range(1, 7)), [])
    return links, anchors
  except Exception:
    raise LinkError(f"Error when reading {filename}")

@functools.lru_cache()
def linklinenumbers(filename, link):
  result = []
  with open(filename, encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
      if link in line:
        result.append(i)
  assert result, (filename, link)
  return result

@functools.lru_cache()
def repo():
  return git.Repo(mainfolder.resolve(strict=True))
@functools.lru_cache()
def blame(filename):
  return repo().blame("HEAD", filename)
@functools.lru_cache()
def is_ancestor(commit1, commit2):
  return repo().is_ancestor(commit1, commit2)

def lastmodified(filename, linenumbers):
  commits = set()
  linecounter = itertools.count(1)
  for commit, lines in blame(filename):
    for line, linenumber in zip(lines, linecounter):
      if linenumber in linenumbers:
        commits.add(commit)

  lastlength = float("inf")
  while len(commits) != lastlength:
    lastlength = len(commits)
    for commit1, commit2 in itertools.permutations(commits, 2):
      if is_ancestor(commit1, commit2):
        commits.remove(commit1)
        break
  return commits

class TestMarkdownLinks(unittest.TestCase):
  def testmarkdownlinks(self):
    for markdownfile in mainfolder.rglob("*.md"):
      markdownfolder = markdownfile.parent
      with self.subTest(markdownfile):
        errors = []
        links, _ = linksandanchors(markdownfile)
        for link in links:
          try:
            dest = link.get("href")
            if dest.startswith("https://"):
              if re.match(r"https://(?:www\.)?github\.com/astropathjhu/astropathpipeline(?:private)?/(?:blob|tree)", dest.lower()):
                raise LinkError(f"Link to {dest}, use a relative link instead")
              continue
            destpath, anchor = re.match("([^#]*)(?:#(.*))?", dest).groups()
            if not destpath:
              destpath = fulldestpath = markdownfile
            else:
              destpath = pathlib.Path(destpath)
              if destpath.is_absolute():
                raise LinkError(f"link to absolute path: {destpath}")
              fulldestpath = (markdownfolder/destpath).resolve()
              try:
                fulldestpath.relative_to(mainfolder)
              except ValueError:
                raise LinkError(f"link to path outside the repo: {dest} (resolves to {fulldestpath})")
            if not fulldestpath.exists():
              raise LinkError(f"link to nonexistent path: {dest} (resolves to {fulldestpath})")
            if anchor is not None:
              if fulldestpath.is_dir():
                fulldestpath = fulldestpath/"README.md"
                if not fulldestpath.exists():
                  raise LinkError(f"link to directory and anchor, but no README.md in the directory: {dest} (resolves to {fulldestpath})")

              if fulldestpath.suffix == ".md":
                try:
                  _, anchors = linksandanchors(fulldestpath)
                except LinkError:
                  raise LinkError(f"link to {dest}, but couldn't parse {fulldestpath}")
                if not any(a.get("id") == anchor for a in anchors):
                  raise LinkError(f"link to nonexistent anchor: {dest} (resolves to {fulldestpath}, couldn't find {anchor})")

              elif fulldestpath.suffix == ".py":
                match = re.match("L([0-9]+)(?:-L([0-9]+))?$", anchor)
                if not match:
                  raise LinkError(f"link to code file {destpath} with anchor {anchor}, expected the anchor to be a github line link e.g. L3 or L5-L7")
                firstline = int(match.group(1))
                if match.group(2) is not None:
                  lastline = int(match.group(2))
                else:
                  lastline = firstline
                with open(fulldestpath) as f:
                  nlines = 0
                  for nlines, line in enumerate(f, start=1):
                    pass
                if firstline > nlines or lastline is not None and lastline > nlines:
                  raise LinkError(f"link to code file {destpath} with anchor {anchor}, but that file only has {nlines} lines")

                codemodified = lastmodified(fulldestpath.relative_to(mainfolder), range(firstline, lastline+1))
                linklines = linklinenumbers(markdownfile, dest)
                markdownmodified = lastmodified(markdownfile.relative_to(mainfolder), linklines)
                for codecommit, markdowncommit in itertools.product(codemodified, markdownmodified):
                  if not is_ancestor(codecommit, markdowncommit):
                    raise LinkError(f"link to code file {destpath} with anchor {anchor} modified in commit {markdowncommit}, but those lines were edited later in {codecommit} - please check and, if it's still ok, modify that line in markdown by adding whitespace or a comment")

              else:
                raise LinkError(f"link to {dest} with an anchor, don't know how to check if an anchor is valid for that file type.")

          except LinkError as e:
            errors.append(e)

        self.assertEqual(len(errors), 0, msg="\n\n"+"\n".join(str(_) for _ in errors))
