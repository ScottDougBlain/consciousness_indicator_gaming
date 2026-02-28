#!/usr/bin/env python3
"""Add ZOIB robustness check slide to Gaming_the_Ghost_Results.pptx."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
import os

PPTX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Gaming_the_Ghost_Results.pptx")

# --- Color palette (matching supplementary section) ---
TITLE_COLOR = RGBColor(0x33, 0x33, 0x33)
BODY_COLOR = RGBColor(0x44, 0x44, 0x44)
CAPTION_COLOR = RGBColor(0x66, 0x66, 0x66)
ACCENT_TEAL = RGBColor(0x2B, 0x8C, 0x8C)
ACCENT_GREEN = RGBColor(0x27, 0xAE, 0x60)
TABLE_HEADER_BG = RGBColor(0x33, 0x33, 0x33)
TABLE_HEADER_FG = RGBColor(0xFF, 0xFF, 0xFF)
BOX_BG = RGBColor(0xF0, 0xF4, 0xF7)
BOX_BORDER = RGBColor(0x2B, 0x8C, 0x8C)
CONCLUSION_BG = RGBColor(0xE8, 0xF5, 0xE9)
CONCLUSION_BORDER = RGBColor(0x27, 0xAE, 0x60)


def set_cell_fill(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag.endswith('solidFill'):
            tcPr.remove(child)
    solidFill = tcPr.makeelement(qn('a:solidFill'), {})
    srgbClr = solidFill.makeelement(qn('a:srgbClr'), {'val': str(color)})
    solidFill.append(srgbClr)
    tcPr.append(solidFill)


def set_cell_border(cell, color_str, width_pt=0.5):
    tcPr = cell._tc.get_or_add_tcPr()
    w = str(int(width_pt * 12700))
    borders_xml = (
        f'<a:tcBorders xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:lnL w="{w}" cmpd="sng"><a:solidFill><a:srgbClr val="{color_str}"/></a:solidFill></a:lnL>'
        f'<a:lnR w="{w}" cmpd="sng"><a:solidFill><a:srgbClr val="{color_str}"/></a:solidFill></a:lnR>'
        f'<a:lnT w="{w}" cmpd="sng"><a:solidFill><a:srgbClr val="{color_str}"/></a:solidFill></a:lnT>'
        f'<a:lnB w="{w}" cmpd="sng"><a:solidFill><a:srgbClr val="{color_str}"/></a:solidFill></a:lnB>'
        f'</a:tcBorders>'
    )
    for child in list(tcPr):
        if child.tag.endswith('tcBorders'):
            tcPr.remove(child)
    tcPr.append(etree.fromstring(borders_xml))


def set_cell_margin(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.set('marL', '45720')
    tcPr.set('marT', '22860')
    tcPr.set('marR', '45720')
    tcPr.set('marB', '22860')


def add_run(para, text, size=Pt(12), bold=False, color=None):
    run = para.add_run()
    run.text = text
    run.font.name = 'Arial'
    run.font.size = size
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    return run


def add_rect(slide, left, top, width, height, fill_color, border_color=None, border_w=Pt(1)):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_w
    else:
        shape.line.fill.background()
    return shape


def create_zoib_slide(prs):
    slide_layout = prs.slide_layouts[0]  # DEFAULT layout
    slide = prs.slides.add_slide(slide_layout)

    # ── TITLE ──
    title_box = slide.shapes.add_textbox(Emu(457200), Emu(137160), Emu(8229600), Emu(457200))
    title_box.text_frame.word_wrap = True
    tp = title_box.text_frame.paragraphs[0]
    tp.alignment = PP_ALIGN.LEFT
    add_run(tp, "Robustness: ZOIB Confirms Core Selectivity (Model 1)",
            size=Pt(24), bold=True, color=TITLE_COLOR)

    # Accent bar
    bar = add_rect(slide, Emu(457200), Emu(594360), Emu(1371600), Emu(27432), ACCENT_TEAL)

    # ── LEFT: Motivation Box ──
    bx_l, bx_t, bx_w, bx_h = Emu(274320), Emu(731520), Emu(2560320), Emu(2103120)
    add_rect(slide, bx_l, bx_t, bx_w, bx_h, BOX_BG, BOX_BORDER, Pt(1))

    # Motivation header
    hdr = slide.shapes.add_textbox(Emu(365760), Emu(777240), Emu(2377440), Emu(274320))
    hdr.text_frame.word_wrap = True
    add_run(hdr.text_frame.paragraphs[0], "Motivation", size=Pt(14), bold=True, color=ACCENT_TEAL)

    # Motivation bullets
    mtx = slide.shapes.add_textbox(Emu(365760), Emu(1051560), Emu(2377440), Emu(1691640))
    mtx.text_frame.word_wrap = True
    bullets = [
        "35.5% of observations at boundaries\n(23.4% at 0, 12.1% at 100)",
        "ZOIB: 3-component model\n(beta + P(0) + P(1))",
        "Bayesian via brms / Stan\n(4 chains \u00d7 4000 iter)",
    ]
    for i, b in enumerate(bullets):
        p = mtx.text_frame.paragraphs[0] if i == 0 else mtx.text_frame.add_paragraph()
        p.space_before = Pt(8) if i > 0 else Pt(0)
        p.space_after = Pt(2)
        add_run(p, "\u2022 ", size=Pt(11), bold=False, color=ACCENT_TEAL)
        add_run(p, b, size=Pt(11), bold=False, color=BODY_COLOR)

    # ── CENTER: Comparison Table ──
    tbl_l, tbl_t = Emu(2926080), Emu(731520)
    tbl_w, tbl_h = Emu(3840480), Emu(2103120)
    rows, cols = 5, 3
    tshape = slide.shapes.add_table(rows, cols, tbl_l, tbl_t, tbl_w, tbl_h)
    tbl = tshape.table

    tbl.columns[0].width = Emu(1554480)
    tbl.columns[1].width = Emu(1143000)
    tbl.columns[2].width = Emu(1143000)

    headers = ["", "LME", "ZOIB"]
    data = [
        ["Direction \u00d7 Type", "F=893.6, p<2e\u207b\u00b9\u2076", "P(\u03b2>0) > 99.9%"],
        ["Inflate raises targets", "+1.2 pp", "+3.0 pp (P=0.995)"],
        ["Suppress lowers targets", "\u221210.7 pp", "\u22126.2 pp (P>0.999)"],
        ["Placebos stable", "<0.3 pp", "<1.5 pp"],
    ]

    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        add_run(p, h, size=Pt(11), bold=True, color=TABLE_HEADER_FG)
        set_cell_fill(cell, TABLE_HEADER_BG)
        set_cell_border(cell, "999999", 0.5)
        set_cell_margin(cell)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            txt_color = RGBColor(0x33, 0x33, 0x33) if ci == 0 else BODY_COLOR
            add_run(p, val, size=Pt(10), bold=(ci == 0), color=txt_color)
            bg = RGBColor(0xF7, 0xF9, 0xFA) if ri % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            set_cell_fill(cell, bg)
            set_cell_border(cell, "CCCCCC", 0.5)
            set_cell_margin(cell)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # ── RIGHT: Convergence Box ──
    cx_l, cx_t, cx_w, cx_h = Emu(6858000), Emu(731520), Emu(2103120), Emu(2103120)
    add_rect(slide, cx_l, cx_t, cx_w, cx_h, BOX_BG, BOX_BORDER, Pt(1))

    chdr = slide.shapes.add_textbox(Emu(6949440), Emu(777240), Emu(1920240), Emu(274320))
    chdr.text_frame.word_wrap = True
    add_run(chdr.text_frame.paragraphs[0], "Convergence", size=Pt(14), bold=True, color=ACCENT_TEAL)

    ctx = slide.shapes.add_textbox(Emu(6949440), Emu(1051560), Emu(1920240), Emu(1691640))
    ctx.text_frame.word_wrap = True
    conv_items = [
        ("\u2713 ", ACCENT_GREEN, "Max R\u0302 = 1.006"),
        ("\u2713 ", ACCENT_GREEN, "0 divergent transitions"),
        ("\u2022 ", ACCENT_TEAL, "zoi = 0.36, coi = 0.34"),
    ]
    for i, (prefix, pcolor, text) in enumerate(conv_items):
        p = ctx.text_frame.paragraphs[0] if i == 0 else ctx.text_frame.add_paragraph()
        p.space_before = Pt(10) if i > 0 else Pt(0)
        p.space_after = Pt(2)
        add_run(p, prefix, size=Pt(12), bold=True, color=pcolor)
        add_run(p, text, size=Pt(11), bold=False, color=BODY_COLOR)

    # ── BOTTOM: Conclusion Box ──
    add_rect(slide, Emu(274320), Emu(2926080), Emu(8686800), Emu(457200),
             CONCLUSION_BG, CONCLUSION_BORDER, Pt(1))
    cbox = slide.shapes.add_textbox(Emu(457200), Emu(2971800), Emu(8320400), Emu(365760))
    cbox.text_frame.word_wrap = True
    cp = cbox.text_frame.paragraphs[0]
    cp.alignment = PP_ALIGN.CENTER
    add_run(cp, "Core selectivity finding robust to ZOIB specification that properly handles boundary mass",
            size=Pt(13), bold=True, color=RGBColor(0x1B, 0x5E, 0x20))

    # ── CAPTION (bottom of slide, matching supplement style) ──
    capbox = slide.shapes.add_textbox(Emu(457200), Emu(4732020), Emu(8229600), Emu(320040))
    capbox.text_frame.word_wrap = True
    add_run(capbox.text_frame.paragraphs[0],
            "Zero-One-Inflated Beta regression (ZOIB) fit with brms (B\u00fcrkner, 2017). "
            "Model: probability ~ direction * indicator_type + (1|model). "
            "Posterior summaries from 16,000 post-warmup draws. All R\u0302 < 1.01.",
            size=Pt(10), bold=False, color=CAPTION_COLOR)

    return slide


def move_slide_to_end_minus(prs, target_index):
    """Move the last slide to the given 0-based index position."""
    sldIdLst = prs.slides._sldIdLst
    children = list(sldIdLst)
    # The new slide is the last child
    new_entry = children[-1]
    sldIdLst.remove(new_entry)
    # Now re-read after removal
    children = list(sldIdLst)
    if target_index >= len(children):
        sldIdLst.append(new_entry)
    else:
        children[target_index].addprevious(new_entry)


def main():
    prs = Presentation(PPTX_PATH)
    print(f"Loaded presentation with {len(prs.slides)} slides")

    new_slide = create_zoib_slide(prs)
    print("Created ZOIB robustness slide")

    # Insert as slide 52 (after current slide 51, 0-indexed = position 51)
    move_slide_to_end_minus(prs, 51)
    print("Positioned slide as #52 (after current slide 51)")

    prs.save(PPTX_PATH)
    print(f"Saved to {PPTX_PATH}")

    # Verify
    verify = Presentation(PPTX_PATH)
    print(f"Verification: {len(verify.slides)} slides total")
    s52 = verify.slides[51]
    for shape in s52.shapes:
        if shape.has_text_frame and shape.text_frame.paragraphs[0].text:
            print(f"  Slide 52 title: \"{shape.text_frame.paragraphs[0].text[:80]}\"")
            break


if __name__ == "__main__":
    main()
