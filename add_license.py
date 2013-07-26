
commdict = {".jam":"#", ".py":"#", ".h":"//", ".cpp":"//", ".in":"#", "":"#", ".txt":""}

rawlic = file("Apache2.0License.txt", "r").readlines()
endtag = "++==++==++==++==++==++==++==++==++==++=="

import os

dirs = ["interfaces/dakota", "tests/dakota"]

for dir in dirs:
    for f in os.listdir(dir):
#        print f
        ext = os.path.splitext(f)
#        print ext[1]
        cc = commdict[ext[1]]
#        print cc
        full = os.path.join(dir,f)
        finlines = file(full).readlines()
        fullnew = "%s.test" % full
        fout = file(fullnew, "w")

        for ln in rawlic:
            fout.write("%s %s" % (cc, ln))
        fout.write("%s %s\n" % (cc, endtag))  ## in case we want to remove the license later
        for ln in finlines:
            fout.write(ln)
        fout.close()
        os.remove(full)
        os.rename(fullnew, full)



