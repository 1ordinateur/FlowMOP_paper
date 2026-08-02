-- Pandoc formatting filter for the FlowMOP revised and tracked manuscripts.
-- It preserves the existing blue HTML revision spans, pairs manuscript
-- captions with their images, and improves table layout without altering the
-- Markdown source documents.

local function stringify(block)
  return pandoc.utils.stringify(block)
end

local function revision_inline(el)
  if el.format ~= "html" then
    return nil
  end

  if el.text:match('^<span%s+style="color:#0066cc">$') then
    return pandoc.RawInline("latex", "{\\color{revisionblue}")
  end

  if el.text == "</span>" then
    return pandoc.RawInline("latex", "}")
  end

  return nil
end

function RawInline(el)
  return revision_inline(el)
end

function Str(el)
  if el.text == "BF₁₀" then
    return {
      pandoc.Str("BF"),
      pandoc.RawInline("latex", "$_{10}$")
    }
  end
  return nil
end

local function image_from_para(block)
  if block and block.t == "Para" and #block.content == 1 and block.content[1].t == "Image" then
    return block.content[1]
  end
  return nil
end

local function is_para(block)
  return block and (block.t == "Para" or block.t == "Plain")
end

local function is_figure_caption(block)
  if not is_para(block) then
    return false
  end
  return stringify(block):match("^Figure%s+%d+%.") ~= nil
    or stringify(block):match("^Figure%s+S%d+[:%.]") ~= nil
end

local function is_short_supplement_label(block)
  if not is_para(block) then
    return false
  end
  return stringify(block):match("^Figure%s+S%d+:?$") ~= nil
end

local function merge_paragraphs(first, second)
  local inlines = pandoc.List()
  for _, inline in ipairs(first.content) do
    inlines:insert(inline)
  end
  inlines:insert(pandoc.Space())
  for _, inline in ipairs(second.content) do
    inlines:insert(inline)
  end
  return pandoc.Para(inlines)
end

local function latex_for_block(block)
  local prepared = block:walk({ RawInline = revision_inline })
  local rendered = pandoc.write(pandoc.Pandoc({ prepared }), "latex", { wrap_text = "none" })
  return rendered:gsub("%s+$", "")
end

local function caption_details(block, kind)
  local text = stringify(block)
  local punctuation = kind == "Figure" and "[:%.]" or ":"
  local label = text:match("^" .. kind .. "%s+([%w]+)" .. punctuation)

  if not label then
    return nil, latex_for_block(block)
  end

  -- Remove only the manual identifier from the rendered caption body. This
  -- works inside Strong/Span wrappers as well, retaining bold and revision-blue
  -- styling while leaving the Wiley class to print FIGURE/TABLE plus the label.
  local caption_tex = latex_for_block(block)
  caption_tex = caption_tex:gsub(kind .. "%s+" .. label .. punctuation .. "%s*", "", 1)
  caption_tex = caption_tex:gsub("%{\\color%{revisionblue%}%}", "")
  return label, caption_tex
end

local function figure_block(image, caption)
  local source = image.src:gsub("\\", "/")
  source = source:gsub("%.svg$", ".png")
  local label, caption_tex = caption_details(caption, "Figure")
  local latex = table.concat({
    "\\begin{figure}[p]",
    "\\centering",
    "\\includegraphics[width=\\linewidth,height=0.82\\textheight,keepaspectratio]{" .. source .. "}",
    "\\begingroup",
    "\\renewcommand{\\thefigure}{" .. label .. "}",
    "\\caption{" .. caption_tex .. "}",
    "\\endgroup",
    "\\end{figure}"
  }, "\n")
  return pandoc.RawBlock("latex", latex)
end

local function is_table_title(block)
  return is_para(block) and stringify(block):match("^Table%s+") ~= nil
end

local function table_start(column_count, label)
  -- The benchmark table is short enough to fit cleanly in portrait once its
  -- columns are allowed to wrap. Start it on a fresh page so its title, note,
  -- and body cannot become detached.
  if label == "1" then
    return pandoc.RawBlock("latex", table.concat({
      "\\clearpage",
      "\\begingroup",
      "\\scriptsize",
      "\\setlength{\\tabcolsep}{2pt}",
      "\\renewcommand{\\arraystretch}{1.10}"
    }, "\n"))
  end

  if column_count >= 6 then
    return pandoc.RawBlock("latex", table.concat({
      "\\clearpage",
      "\\begin{landscape}",
      "\\begingroup",
      "\\scriptsize",
      "\\setlength{\\tabcolsep}{3pt}",
      "\\renewcommand{\\arraystretch}{1.12}"
    }, "\n"))
  end

  return pandoc.RawBlock("latex", table.concat({
    "\\begingroup",
    "\\small",
    "\\setlength{\\tabcolsep}{4pt}",
    "\\renewcommand{\\arraystretch}{1.14}"
  }, "\n"))
end

local function table_end(column_count, label)
  if column_count >= 6 and label ~= "1" then
    return pandoc.RawBlock("latex", "\\endgroup\n\\end{landscape}\n\\clearpage")
  end
  return pandoc.RawBlock("latex", "\\endgroup")
end

local function table_caption(block, continued)
  local label, caption_tex = caption_details(block, "Table")
  if continued then
    caption_tex = caption_tex .. " (continued)"
  end

  return pandoc.RawBlock(
    "latex",
    table.concat({
      "\\captionsetup{type=table}",
      "\\begingroup",
      "\\renewcommand{\\thetable}{" .. label .. "}",
      "\\captionof{table}{" .. caption_tex .. "}",
      "\\endgroup"
    }, "\n")
  )
end

local function set_table_widths(table_block)
  local column_count = #table_block.colspecs
  if column_count < 5 then
    return
  end

  -- Pandoc otherwise emits natural-width l/r columns, which makes long
  -- manuscript headers run beyond the page. Explicit proportional widths let
  -- the cells wrap within Wiley's portrait or landscape text block.
  local width = 0.96 / column_count
  for index, colspec in ipairs(table_block.colspecs) do
    table_block.colspecs[index] = { colspec[1], width }
  end
end

local function add_table(output, table_block, caption_block, note_block, continued)
  local column_count = #table_block.colspecs
  local label = caption_block and select(1, caption_details(caption_block, "Table")) or nil
  set_table_widths(table_block)
  output:insert(table_start(column_count, label))
  if caption_block then
    output:insert(table_caption(caption_block, continued))
  end
  if note_block then
    output:insert(note_block)
  end
  output:insert(table_block)
  output:insert(table_end(column_count, label))
end

function Pandoc(doc)
  local output = pandoc.List()
  local in_references = false
  local body_start = 1
  local acknowledgements_index = nil
  local funding_index = nil
  local abstract_index = nil
  local introduction_index = nil
  local contact_index = nil

  for index, block in ipairs(doc.blocks) do
    local text = stringify(block)
    if block.t == "Header" and block.level == 2 and text == "Acknowledgements" then
      acknowledgements_index = index
    elseif block.t == "Header" and block.level == 2 and text == "Abstract" then
      abstract_index = index
    elseif block.t == "Header" and block.level == 2 and text == "Introduction" then
      introduction_index = index
    elseif is_para(block) and text == "Funding information:" then
      funding_index = index
    elseif is_para(block) and text == "Contact information for all authors:" then
      contact_index = index
    end
  end

  if abstract_index and introduction_index then
    local abstract_blocks = pandoc.List()
    for index = abstract_index + 1, introduction_index - 1 do
      abstract_blocks:insert(doc.blocks[index])
    end
    doc.meta.abstract = pandoc.MetaBlocks(abstract_blocks)
    body_start = introduction_index
  end

  if contact_index and acknowledgements_index then
    local contact_blocks = pandoc.List()
    for index = contact_index + 1, acknowledgements_index - 1 do
      contact_blocks:insert(doc.blocks[index])
    end
    doc.meta.contactinfo = pandoc.MetaBlocks(contact_blocks)
  end

  if acknowledgements_index and funding_index then
    local acknowledgements_blocks = pandoc.List()
    for index = acknowledgements_index + 1, funding_index - 1 do
      acknowledgements_blocks:insert(doc.blocks[index])
    end
    doc.meta.acknowledgements = pandoc.MetaBlocks(acknowledgements_blocks)
  end

  if funding_index and abstract_index then
    local funding_blocks = pandoc.List()
    for index = funding_index + 1, abstract_index - 1 do
      funding_blocks:insert(doc.blocks[index])
    end
    doc.meta.funding = pandoc.MetaBlocks(funding_blocks)
  end

  local i = body_start

  while i <= #doc.blocks do
    local block = doc.blocks[i]
    local heading = block.t == "Header" and stringify(block) or nil
    local source_level = block.t == "Header" and block.level or nil

    -- The Markdown title is a level-one heading, so body headings begin at
    -- level two. The title is moved into Wiley's front matter above; promote
    -- the remaining headings so Introduction, Methods, and peers are sections.
    if source_level and source_level >= 2 then
      block.level = source_level - 1
    end

    if source_level == 2 then
      output:insert(pandoc.RawBlock("latex", "\\FloatBarrier"))
      if heading == "Supplementary data" or heading == "References" then
        output:insert(pandoc.RawBlock("latex", "\\clearpage"))
      end
    end

    if source_level == 2 and heading == "References" then
      output:insert(block)
      output:insert(pandoc.RawBlock("latex", "\\begin{hangparas}{1.5em}{1}"))
      in_references = true
      i = i + 1
      goto continue
    end

    -- Supplementary figures S3/S4 use a short label paragraph, a descriptive
    -- paragraph, and then the image.
    if is_short_supplement_label(block)
      and is_para(doc.blocks[i + 1])
      and image_from_para(doc.blocks[i + 2]) then
      output:insert(figure_block(
        image_from_para(doc.blocks[i + 2]),
        merge_paragraphs(block, doc.blocks[i + 1])
      ))
      i = i + 3
      goto continue
    end

    -- Supplementary figures S1/S2 place the caption before the image.
    if is_figure_caption(block) and image_from_para(doc.blocks[i + 1]) then
      output:insert(figure_block(image_from_para(doc.blocks[i + 1]), block))
      i = i + 2
      goto continue
    end

    -- Main figures place the image before the complete caption paragraph.
    local image = image_from_para(block)
    if image and is_figure_caption(doc.blocks[i + 1]) then
      output:insert(figure_block(image, doc.blocks[i + 1]))
      i = i + 2
      goto continue
    end

    -- Manuscript table titles become numbered captions. A single explanatory
    -- paragraph between a title and table is kept with that table (Table 1).
    if is_table_title(block) then
      local table_index = i + 1
      local note_block = nil
      if is_para(doc.blocks[table_index])
        and doc.blocks[table_index + 1]
        and doc.blocks[table_index + 1].t == "Table" then
        note_block = doc.blocks[table_index]
        table_index = table_index + 1
      end

      if doc.blocks[table_index] and doc.blocks[table_index].t == "Table" then
        add_table(output, doc.blocks[table_index], block, note_block, false)
        table_index = table_index + 1

        -- Table S1A is divided into consecutive blocks for readable column
        -- widths. Give each continuation the same visible table identifier.
        while doc.blocks[table_index] and doc.blocks[table_index].t == "Table" do
          add_table(output, doc.blocks[table_index], block, nil, true)
          table_index = table_index + 1
        end

        i = table_index
        goto continue
      end
    end

    if block.t == "Table" then
      add_table(output, block, nil, nil, false)
      i = i + 1
      goto continue
    end

    output:insert(block)
    i = i + 1

    ::continue::
  end

  if in_references then
    output:insert(pandoc.RawBlock("latex", "\\end{hangparas}"))
  end

  doc.blocks = output
  return doc
end
