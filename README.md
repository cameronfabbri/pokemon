# PokeGAN

To set up environment: `conda create -n tf tensorflow-gpu cudatoolkit=10.1`

### Data

#### Naming Convention
`id_game_direction_shiny_icon_mega_num` where each is `int_str_str_bool_bool_bool_int`.

`id`: National Pokedex ID<br>
`game`: The game the sprite is from<br>
`direction`: Can be `front`, `back`, `left` or `right`<br>
`shiny`: Either true `t` or false `f` <br>
`icon`: Either true `t` or false `f` for whether or not it's an icon <br>
`mega`: Either true `t` or false `f`<br>
`num`: Some games have animated sprites, so this number corresponds to each
frame of the animation.

#### Game abreviations
`green`: Green version
`rb`: Red/Blue version


### Ideas

#### Data

    - Go through pokemon sprites and delete excact duplicates that may occur due to looping gif
    - Find most common color and slightly change it

#### StyleGAN
    - Pretrain on large dataset of other sprites

#### Style Transfer
    - Different generations/art styles
    - Different types (fire, grass, etc.)
    - Different "shape" (bird, four legs, etc.)


