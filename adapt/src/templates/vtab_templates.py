from .utils import append_proper_article, get_plural

flowers_template = [
    lambda c: f"a photo of a {c}, a type of flower.",
    lambda c : f"a photo of {append_proper_article(c)}, a type of flower.",
]

stanfordcars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of {append_proper_article(c)}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

aircraft_template = [
    lambda c: f'a photo of a {c}, a type of aircraft.',
    lambda c: f'a photo of the {c}, a type of aircraft.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of {append_proper_article(c)} texture.',
    lambda c: f'a photo of {append_proper_article(c)} pattern.',
    lambda c: f'a photo of {append_proper_article(c)} thing.',
    lambda c: f'a photo of {append_proper_article(c)} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]