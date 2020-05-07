import os
import re


def main():

    # `id_game_direction_shiny_icon_mega_num` where each is `int_str_str_bool_bool_bool_int`.
    game = 'bw_ani'

    d = os.path.join('data', 'pokemon', game)

    files = sorted([os.path.join(d, x) for x in os.listdir(d)])
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    direction = 'front'
    shiny = 'f'
    icon = 'f'
    mega = 'f'
    num = '1'

    pid = 1
    game = 'bw'

    for f in files:
        #new_name = str(pid)+'_'+game+'_'+direction+'_'+shiny+'_'+icon+'_'+mega+'_'+num+'.png'
        #new_name = os.path.join(d, new_name)
        #os.rename(f, new_name)

        #new_name = str(pid)+'_'+game+'_'+direction+'_'+shiny+'_'+icon+'_'+mega+'.gif'
        #new_name = os.path.join(d, os.path.basename(new_name))
        #os.rename(f, new_name)
        command = 'convert -coalesce '+f+' '+f.replace('.gif','.png')
        os.system(command)
        print(command)
        #print(f,'-->',new_name)
        pid += 1


if __name__ == '__main__':
    main()
