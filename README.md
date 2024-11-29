# MSuAI
Unfinished project

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

U fajlu example1.py obradjene su prve dvije tacke:
    * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    * Apply a distortion correction to raw images.

U fajlu example2.py obradjena je treca tacka:
    * Use color transforms, gradients, etc., to create a thresholded binary image.

U fajlu example3.py obradjena je cetvrta tacka:
    * Apply a perspective transform to rectify binary image ("birds-eye view").

U fajlu example1.py obradjene su tacke 5 i 7:
    * Detect lane pixels and fit to find the lane boundary.
    * Warp the detected lane boundaries back onto the original image.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Trebamo generisati 3D koordinate tacaka u realnom svijetu koje odgovaraju uglovima sahovske table.
objp = np.zeros((6*9, 3), np.float32) => stvara 3D mrezu tacaka sa 54 tacke odnosno 6 redova i 9 kolona.
np.mgrid[0:9, 0:6] => ova komanda generise koordinate u 2D prostoru za redove i kolone dok T.reshape(-1, 2) pretvara 2D mrezu u jednodimenzionalni niz sa parovima koordinata.

Zatim pravimo liste za skladistenje tacaka:
objpoints = []  => lista koja ce cuvati koordinate objektnih tacaka (sve objp)
imgpoints = []  => lista za slikovne tacke odnosno koordinate uglova sahovske table pronadjene na slikama

Dolazimo do for petlje. Ovde ucitavamao sve slike sahovske table. 
Konvertujemo sliku u sivu skalu pomocu ove funkcije "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)" zato sto funkcija za detekciju uglova radi sa jednokanalnim slikama.
Pomocu funckije "cv2.findChessboardCorners" detektujemo uglove na sahovskoj tabli. Ova funkcija vraca True/False vrijednost u zavisnosti da li je pronadjen ugao ili ne i vraca njihove koordinate.
Ako je ret == True onda ubacuje objp u objpoints i corners u imgpoints.

Nakon sto smo sve to zavrsili pomocu funckije cv2.drawChessboardCorners crtamo sve pronadjene uglove na slici prikazujemo svaku sliku pomocu cv2.imshow funkcije u odredjenom vremenskom trajanju.

Zatim dolazimo do kalibracije kamere: ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

Ova funkcija prima sledece parametre: 
objpoints => Lista 3D objektnih tacaka
imgpoints => Lista 2D tacaka sa slika
gray.shape[::-1] => Velicina slike u formatu širina/visina

Izlazni parametri:
ret => Indikator uspjeha kalibracije
mtx => Matrica kamere
dist => Koeficijenti distorzije
rvecs => Rotacioni vektori za svaku sliku
tvecs => Translacioni vektori za svaku sliku

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Pomocu funckije cv2.imread ucitavamo sliku sa koje zelimo ukloniti distorziju (Prilazem sliku calibration2.jpg)
Pomocu funkcije cv2.undistort ispravljamo distorziju (Prilazem sliku undistorted.jpg). Ova funkcija prima parametre:
img => Ulazna slikua
mtx => Matrica kamere.
dist => Koeficijent distorzije

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

U ovom dijelu programa koristimo kombinaciju Sobel filtera i analizu boja za detekciju ivica. Rezultat je binarna slika koja oznacava oblasti od interesa kao sto su linije na putu.

Funkcija abs_sobel_tresh detektuje ivice na slici koristeci Sobel filter.
Ulazna slika img se konvertuje u grey scale sliku radi lakse obrade. To radimo pomocu funkcije "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"
Parametar orient nam odredjuje u kom ce se pravcu primjenjivati Sobel filter, da li u horizontalnom (orient = 'x') ili vertikalnom (orient = 'y')
Da bismo izbjegli negativne vrijednosti koristimo funkciju np.absolute(sobel) i smjestamo u promjenjivu abs_sobel.
Rezultat se skalira u opsegu 0-255.
Primjenjujemo treshold, vrijednosti u opsgu tresh_min do tresh_max postaju 1, a sve ostale vrijednosti postaju 0. Kao povratnu vrijednost dobijamo binarnu sliku.

Zatim dolazimo do funckije color_treshold koja identifikuje oblasti na slici prema intenzitetu boje.
Ulazna slika img se konvertuje iz RGB u HLS prostor boja. Izdvaja se s kanal (s_channel) koji predstavlja zasicenost boja (saturation).
Pikseli u opsegu treshholda 170-255 postaju 1 dok ostali pikseli postaju 0. Povratna vrijednost je binarna slika.

Da bismo dobili konacnu binarnu sliku kombinujemo ova dva rezultata kako bismo preciznije dobili oblasti od interesa na binarnoj slici.
Sobel filter smjestamo u grad_binary, a color_treshold u color_binary.
Kreira se binarna slika istih dimenzija kao grad_binary i color_binary.
Pikseli postaju 1 ako su 1 u bilo grad_binary ili color_binary.
Mnozimo binarnu sliku sa 255 kako bi pikseli sa vrednošću 1 postali bijele boje (255), a pikseli sa vrednošću 0 ostali crni (0).

Kao rezulat rada programa prilazem ulaznu sliku "test4.jpg" i izlaznu sliku "binary_output.jpg"

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!
