-- Keep response-letter tables within the text block by assigning explicit
-- proportional widths. Pandoc otherwise emits natural-width columns, which
-- do not wrap long reviewer comments or coverage-index labels.

local function tracked_inline(el)
  if el.format ~= "html" then
    return nil
  end
  if el.text:match('^<span%s+style="color:#0066cc">$') then
    return pandoc.RawInline("latex", "{\\color{revisionblue}")
  end
  if el.text:match('^<span%s+style="color:#c00000">$') then
    return pandoc.RawInline("latex", "{\\color{revisionred}")
  end
  if el.text == "</span>" then
    return pandoc.RawInline("latex", "}")
  end
  if el.text == "<s>" then
    return pandoc.RawInline("latex", "\\sout{")
  end
  if el.text == "</s>" then
    return pandoc.RawInline("latex", "}")
  end
  return nil
end

function RawInline(el)
  return tracked_inline(el)
end

function Table(table_block)
  local column_count = #table_block.colspecs
  if column_count == 0 then
    return nil
  end

  if column_count == 2 then
    table_block.colspecs[1] = { table_block.colspecs[1][1], 0.64 }
    table_block.colspecs[2] = { table_block.colspecs[2][1], 0.32 }
  else
    local width = 0.96 / column_count
    for index, colspec in ipairs(table_block.colspecs) do
      table_block.colspecs[index] = { colspec[1], width }
    end
  end

  return table_block
end
