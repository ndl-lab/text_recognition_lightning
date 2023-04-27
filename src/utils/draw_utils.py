from pathlib import Path
import functools
import difflib

from PIL import Image, ImageDraw, ImageFont


def get_escaped_char(sa1, sb1):
    if sa1 is None:
        return '', sb1, ''
    if sa1 == sb1:
        return sa1, sb1, 1

    sm = difflib.SequenceMatcher(None, sa1, sb1)
    sa2 = str()
    sb2 = str()
    for tag, ia1, ia2, ib1, ib2 in sm.get_opcodes():
        if tag == 'equal':
            sa2 += "\033[0m"
            sb2 += "\033[0m"
        elif tag == 'replace':
            sa2 += "\033[31;1m"
            sb2 += "\033[31;1m"
        elif tag == 'insert':
            sb2 += "\033[33;1m"
        elif tag == 'delete':
            sa2 += "\033[33;1m"
        sa2 += sa1[ia1:ia2]
        sb2 += sb1[ib1:ib2]
    sa2 += '\033[0m'
    sb2 += '\033[0m'
    return sa2, sb2, sm.ratio()


class LineImageDraw():
    def __init__(self, font_path: str, output_dir: str = None, height: int = 32, draw_all: bool = False):
        if font_path is None:
            raise AttributeError("need valid <font_path>")
        self.font_path = Path(font_path)
        if not self.font_path.exists():
            raise FileNotFoundError(font_path)
        assert self.font_path.exists()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.height = height
        self.draw_all = draw_all

        dtmp = ImageDraw.Draw(Image.new('L', (height, height)))
        self._font = ImageFont.truetype(font_path, height)
        self._textsize = functools.partial(dtmp.textbbox, font=self._font, xy=(0, 0))

        self.idx = 0

    def draw_escape_colored_text(self, text, draw, pos=[0, 0]):
        it = iter(text)
        color = (255, 255, 255)  # color
        for c in it:
            if c == '\033':  # begin color code
                n = next(it)
                while n[-1] != 'm':  # end color code '\033m'
                    n += next(it)
                if n == '[0m':
                    color = (255, 255, 255)
                elif n == '[31;1m':
                    color = (255, 0, 0)
                elif n == '[33;1m':
                    color = (255, 255, 0)
                continue
            else:
                size = self._textsize(text=c)[2:]

            draw.text(pos, c, font=self._font, fill=color)
            pos[0] += size[0]

    def __call__(self, image, target, pred, pid):
        if not self.draw_all and target == pred:
            return

        w1, h1 = self._textsize(text=target)[2:]
        w2, h2 = self._textsize(text=pred)[2:]
        target, pred, r = get_escaped_char(target, pred)

        width = round(image.width * self.height / image.height)
        image = image.resize((width, self.height))

        canvas = Image.new('RGB', (max(image.width, w1, w2), image.height + h1 + h2), (20, 20, 50))

        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        self.draw_escape_colored_text(target, draw, [0, image.height])
        self.draw_escape_colored_text(pred, draw, [0, image.height + h1])

        canvas.save(self.output_dir / f"{pid}-{self.idx:09d}-{r:.2f}.png")
        self.idx += 1


def get_render(task, font_path=None, **kwargs):
    if task == "render":
        return LineImageDraw(font_path=font_path, **kwargs)
    elif task == "xml":
        return get_escaped_char
