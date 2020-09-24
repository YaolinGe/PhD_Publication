print("hello world, new test")



def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix


index_html = open("../index.htm", 'r')
contents = index_html.readlines()
# print(contents)
new_file = open("../index.htm", 'w+')

image_path = 'Path_planning_BayesOpt_smart'
number = 0
nr = 'Figs_' + str(number)


for i in contents:
	if i == "</body>\n": 
		break
	new_file.writelines(i)

write_file = open("../index.htm", 'a+')
objects = ["mean", "std", "EI"]

for i in range(10):
	write_file.writelines("<h3> The {} realisation </h3>\n".format(make_ordinal(i)))
	for j in objects:
		object_path_str = "<img src=\"Path_planning_BayesOpt_smart/" + nr + "/fig_" + str(i) + "/" + j + ".gif\""
		object_alt_str = " alt=" + "\"Posterior " + j + " variation with updated path steps\""
		object_img_str = " width=" + "\"500\"" + "height=" + "\"400\"" + ">\n"
		write_file.writelines(object_path_str + object_alt_str + object_img_str)
	write_file.writelines("<hr>\n")


write_file.writelines("</body>\n")
write_file.writelines("</html>\n")


# <h3> The 1st realisation </h3>
# <!-- <a href="https://www.apple.com">This is Apple</a>
# <h5>Thi sis the 5th title</h5>
# <h1>thi is is much bitter </h1> -->

# <!-- <p title = 'Steven Jobs'></p>
# <p title = "Yaolin Ge"></p> -->



new_file.close()
index_html.close()
write_file.close()

