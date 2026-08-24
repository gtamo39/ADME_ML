"""Shared styling helpers for the publication .pptx slides (python-pptx)."""
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


NO_GRID = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'   # built-in "No Style, No Grid" table style

# shared "light/navy" table palette (used by the potency + datasets slides)
# BORDER 5B9BD5 = user's preferred inter-row + outer horizontal-rule color (2026-07-20)
NAVY, MUTED, HEADER_BG, BORDER = '1F3A6B', '6B7A99', 'EEF2F8', '5B9BD5'


def hex_color(h):
    return RGBColor.from_string(h.lstrip('#'))


def plain_table(tbl):
    """Strip the default table style (which draws its own banding + white grid borders): set the
    'No Style, No Grid' style + disable first-row/banding, so only explicitly-set cell fills and
    rules are drawn. Call right after creating the table."""
    tbl.first_row = tbl.horz_banding = False
    tblPr = tbl._tbl.find(qn('a:tblPr'))
    if tblPr is None:
        tblPr = parse_xml(f'<a:tblPr {A}/>'); tbl._tbl.insert(0, tblPr)
    for sid in tblPr.findall(qn('a:tableStyleId')):
        tblPr.remove(sid)
    tblPr.append(parse_xml(f'<a:tableStyleId {A}>{NO_GRID}</a:tableStyleId>'))


def horizontal_rules(cell, color='5B9BD5', w=9525):
    """Horizontal-only cell borders: top+bottom ruled in `color`, left+right explicitly removed
    (overrides the table style so NO vertical column separators are drawn). `w` in EMU (9525 = 0.75pt)."""
    color = color.lstrip('#')
    tcPr = cell._tc.get_or_add_tcPr()
    tag = {'L': 'a:lnL', 'R': 'a:lnR', 'T': 'a:lnT', 'B': 'a:lnB'}
    solid = lambda t: f'<{t} {A} w="{w}" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:prstDash val="solid"/></{t}>'
    none = lambda t: f'<{t} {A} w="0"><a:noFill/></{t}>'
    # insert reversed so final child order is L, R, T, B (schema order), before any fill element
    for side, xml in reversed([('L', none), ('R', none), ('T', solid), ('B', solid)]):
        t = tag[side]
        for e in tcPr.findall(qn(t)):
            tcPr.remove(e)
        tcPr.insert(0, parse_xml(xml(t)))
