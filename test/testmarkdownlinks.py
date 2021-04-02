import bs4, functools, marko.ext.toc, pathlib, re, slugify, unittest

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

class LinkError(Exception): pass

@functools.lru_cache()
def linksandanchors(filename):
  try:
    with open(filename) as f:
      markdown = f.read()
    html = marko.Markdown(extensions=[GithubToc()])(markdown)
    soup = bs4.BeautifulSoup(html, features="lxml")
    links = soup.findAll("a", attrs={"href": re.compile(".*")})
    anchors = sum((soup.findAll(f"h{i}", attrs={"id": re.compile(".*")}) for i in range(1, 7)), [])
    #import pprint; pprint.pprint(anchors); input()
    return links, anchors
  except Exception:
    raise LinkError(f"Error when reading {filename}")

class TestMarkdownLinks(unittest.TestCase):
  def testmarkdownlinks(self):
    mainfolder = (thisfolder/"..").resolve()
    for markdownfile in mainfolder.rglob("*.md"):
      markdownfolder = markdownfile.parent
      with self.subTest(markdownfile):
        errors = []
        links, _ = linksandanchors(markdownfile)
        for link in links:
          try:
            dest = link.get("href")
            if dest.startswith("https://"): continue
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
              try:
                _, anchors = linksandanchors(fulldestpath)
              except LinkError:
                raise LinkError(f"link to {dest}, but couldn't parse {fulldestpath}")
              if not any(a.get("id") == anchor for a in anchors):
                raise LinkError(f"link to nonexistent anchor: {dest} (resolves to {fulldestpath}, couldn't find {anchor})")
          except LinkError as e:
            errors.append(e)

        self.assertEqual(len(errors), 0, msg="\n\n"+"\n".join(str(_) for _ in errors))
