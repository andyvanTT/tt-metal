
vim.g.mapleader = " "

vim.opt.number = true
vim.opt.relativenumber = true
vim.opt.hlsearch = true
vim.opt.ignorecase = true
vim.opt.smartcase = true
vim.opt.tabstop = 2
vim.opt.shiftwidth = 2
vim.opt.expandtab = true
vim.opt.wrap = false
vim.opt.mouse = "a"


vim.keymap.set("n", "p", "<Cmd>lua Paste_after_and_trim()<CR>")
vim.keymap.set("n", "P", "<Cmd>lua Paste_before_and_trim()<CR>")
vim.keymap.set({'n', 'v'}, '<Leader>l', ':tabnext<CR>',     { silent = true })
vim.keymap.set({'n', 'v'}, '<Leader>h', ':tabprevious<CR>', { silent = true })
