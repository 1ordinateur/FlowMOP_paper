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

local function figure_block(image, caption)
  local source = image.src:gsub("\\", "/")
  source = source:gsub("%.svg$", ".png")
  local caption_tex = latex_for_block(caption)
  local latex = table.concat({
    "\\begin{figure}[p]",
    "\\centering",
    "\\includegraphics[width=\\linewidth,height=0.82\\textheight,keepaspectratio]{" .. source .. "}",
    "\\caption*{" .. caption_tex .. "}",
    "\\end{figure}"
  }, "\n")
  return pandoc.RawBlock("latex", latex)
end

local function is_table_title(block)
  return is_para(block) and stringify(block):match("^Table%s+") ~= nil
end

local function table_start(column_count)
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

local function table_end(column_count)
  if column_count >= 6 then
    return pandoc.RawBlock("latex", "\\endgroup\n\\end{landscape}\n\\clearpage")
  end
  return pandoc.RawBlock("latex", "\\endgroup")
end

local function table_caption(block)
  return pandoc.RawBlock(
    "latex",
    "\\captionsetup{type=table}\n\\captionof*{table}{" .. latex_for_block(block) .. "}"
  )
end

local function add_table(output, table_block, caption_block)
  local column_count = #table_block.colspecs
  output:insert(table_start(column_count))
  if caption_block then
    output:insert(table_caption(caption_block))
  end
  output:insert(table_block)
  output:insert(table_end(column_count))
end

function Pandoc(doc)
  local output = pandoc.List()
  local in_abstract = false
  local in_references = false
  local i = 1

  while i <= #doc.blocks do
    local block = doc.blocks[i]
    local heading = block.t == "Header" and stringify(block) or nil

    -- The title is supplied through template metadata.
    if i == 1 and block.t == "Header" and block.level == 1 then
      i = i + 1
      goto continue
    end

    if block.t == "Header" and block.level == 2 and heading == "Abstract" then
      output:insert(pandoc.RawBlock("latex", "\\begin{abstract}"))
      in_abstract = true
      i = i + 1
      goto continue
    end

    if in_abstract and block.t == "Header" and block.level == 2 then
      output:insert(pandoc.RawBlock("latex", "\\end{abstract}"))
      in_abstract = false
    end

    if block.t == "Header" and block.level == 2 then
      output:insert(pandoc.RawBlock("latex", "\\FloatBarrier"))
      if heading == "Supplementary data" or heading == "References" then
        output:insert(pandoc.RawBlock("latex", "\\clearpage"))
      end
    end

    if block.t == "Header" and block.level == 2 and heading == "References" then
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

    -- Bold manuscript table titles become proper unnumbered table captions.
    if is_table_title(block) and doc.blocks[i + 1] and doc.blocks[i + 1].t == "Table" then
      add_table(output, doc.blocks[i + 1], block)
      i = i + 2
      goto continue
    end

    if block.t == "Table" then
      add_table(output, block, nil)
      i = i + 1
      goto continue
    end

    output:insert(block)
    i = i + 1

    ::continue::
  end

  if in_abstract then
    output:insert(pandoc.RawBlock("latex", "\\end{abstract}"))
  end
  if in_references then
    output:insert(pandoc.RawBlock("latex", "\\end{hangparas}"))
  end

  doc.blocks = output
  return doc
end
