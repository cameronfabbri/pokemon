# PokeGAN

To set up environment: `conda create -n tf tensorflow-gpu cudatoolkit=10.1`

### Data

pokemon naming convention: `id_game_direction_shiny_mini_mega_num` where each is `int_str_str_bool_bool_bool_num`.

`id`: National Pokedex ID
`game`: The game the sprite is from
`direction`: Can be `front`, `back`, `left` or `right`
`shiny`: Either true `t` or false `f`
`mega`: Either true `t` or false `f`
`num`: Some games have animated sprites, so this number corresponds to each
frame of the animation.

### Ideas

#### StyleGAN
    - Pretrain on large dataset of other sprites

#### Style Transfer
    - Different generations/art styles
    - Different types (fire, grass, etc.)
    - Different "shape" (bird, four legs, etc.)


### TODO

