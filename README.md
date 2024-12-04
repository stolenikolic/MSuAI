### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

U fajlu calibration.py obradjene su prve dvije tacke:
    * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    * Apply a distortion correction to raw images.

U fajlu binary_output.py obradjena je treca tacka:
    * Use color transforms, gradients, etc., to create a thresholded binary image.

U fajlu birds_eye_perspective.py obradjena je cetvrta tacka:
    * Apply a perspective transform to rectify binary image ("birds-eye view").

U fajlu lane_detection.py je obradjena peta tacka:
    * Detect lane pixels and fit to find the lane boundary.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

#########################
calibration.py
#########################

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

#########################
binary_output.py
#########################

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

Kao rezulat rada programa prilazem ulaznu sliku "test6.jpg" i izlaznu sliku "binary_output.jpg"

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

#########################
birds_eye_perspective.py
#########################

Ovaj kod je namijenjen primjeni perspektivne transformacije na binarne slike, kako bi se slika prebacila u bird's-eye view perspektivu (pogled odozgo). 
Koristi se u obradi slike, posebno u zadacima kao sto je detekcija linija na putu.

Funkcija perspective_transform_binary(binary_img) obavlja perspektivnu transformaciju na binarnu sliku.
Ulazni parametar je binary_img odnsno binarna, 2D slika koja treba da bude transformisana.
Izlazni parametri su (funkcija vraca) "warped" sto je rezultat prespektivne transformacije slike u pticiju prespektivu i "M" matrica perspektivne transformacije 3x3.
Definisu se dimenzije slike za transformaciju pomocu binary_img.shape[1], binary_img.shape[0].
Postavljaju se tacke izvora (src) na originalnoj slici. Ove tacke definisu pravougaoni trapez na slici.
Postavljaju se tacke destinacije za transformaciju. Tacke su pravougaoni prostor u birds eye perspektivi.
Izracunava se matrica transformacije pomocu funkcije cv2.getPerspectiveTransform(src, dst)
Primjenjuje se perspektivna transformacija pomocu cv2.warpPerspective(). 

Funkcija bird_eye(binary_image_frame) je namjenjena kako bismo kasnije mogli pozivati ovu transformaciju iz drugih fajlova.
Npr kada budemo radili obradjivanje frejmova sa video snimka, mozemo iz tog fajla direktno da pozivamo ovu funkciju i da trenutni frejm prebacimo u birds eye perspektivu.
Zbog toga funkcija prvo pokusava da ulaznu sliku dobije preko putanje koju smo mi odredili (ukoliko pokrecemo samo ovaj fajl) ili da ulazna slika bude frejm sa video snimka ukoliko je pozivamo iz drugog fajla.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Ovo sam radio u sledecem fajlu, bice objasnjeno ispod.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#########################
lane_detection.py
#########################

U ovom fajlu implementiramo proces detekcije linija traka na binarnim slikama u birds eye prespektivi i njihovu vizuelizaciju.

detect_lane_pixels_and_fit_poly(binary_warped)
Ova funkcija detektuje piksele traka koristeci histogram binarne slike i fituje polinome drugog reda za leve i desne granice traka.
Ulazni parametar:
binary_warped => Binarna slika (2D niz) sa perspektivnom transformacijom (bird's-eye view).

Izlazni parametri:
left_fit => Koeficijenti polinoma drugog reda za levu traku.
right_fit => Koeficijenti polinoma drugog reda za desnu traku.

Prvo radimo histogram. Histogram se koristi za inicijalnu detekciju gustine piksela i omogucava pronalazak pocetnih tacaka. Horizontalna suma donje polovine slike koristi se za identifikaciju piksel-gustine. Histogram se deli na levo i desno kako bi se odredile početne pozicije traka.
Zatim na red dolaze sliding windows. Slika se deli na n prozora (određeno parametrom nwindows). Prozori se pomeraju vertikalno i horizontalno da bi se identifikovali pikseli traka.
Fitovanje polinoma: Pikseli detektovani u prethodnom koraku koriste se za fitovanje polinoma drugog reda za lijeve i desne granice traka.

draw_lane_lines_on_original(binary_warped, left_fit, right_fit)
Cilj ove funckije je iscrtavanje detektovanih linija plavom i crvenom bojom.
Ulazni parametri:
binary_warped => Binarna slika na kojoj se crtaju linije.
left_fit => Koeficijenti polinoma za levu traku.
right_fit => Koeficijenti polinoma za desnu traku.
Izlazni parametar je out_img, odnosno slika sa nacrtanim linijama u plavoj i crvenoj boji.
Prvo kopiramo binarnu sliku u RGB format kako bismo omogucili crtanje u boji.
Generisu se y vrijednosti ravnomjerno rasporedjene duz visine slike.
Izracunava se x vrijednost za lijevu i desnu traku koristeci koeficijente polinoma.
Pomocu cv2.line() iscrtavamo linije u odgovarajucim bojama, plava za lijevu traku i crvena za desnu traku.

lane_detection(binary_warped_frame)
Ova funkcija takodje sluzi samo za poziv ovog programa iz drugog fajla kada budemo obradjivali frejmove sa video snimaka.

Primjer: output_images/lane_lines_image.jpg

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

#########################
back_to_original_img.py
#########################

Ovaj program implementira proces prepoznavanja traka (lane detection) na binarnoj slici u "bird's-eye" perspektivi i preklapanje detektovanih traka na originalnu sliku. Koristi se za zadatke poput asistencije u vožnji (npr. autonomna vozila).

warp_lane_to_original(binary_warped, original_img, left_fit, right_fit, Minv)
Ulazni parametri:
binary_warped => binarna slika sa detektovanim trakama.
original_img => originalna slika u RGB formatu.
left_fit, right_fit => koeficijenti parabola za levu i desnu traku.
Minv => matrica inverzne perspektivne transformacije.
Izlaz je slika sa preklopljenim trakama na originalnu perspektivu.

Generisanje tačaka za leve i desne parabole:
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]])

Crtanje traka na bird's-eye slici:
cv2.fillPoly(lane_img, np.int_([pts]), (0, 255, 0))

Transformacija nazad na originalnu perspektivu:
lane_on_original = cv2.warpPerspective(lane_img, Minv, (original_img.shape[1], original_img.shape[0]))

Kombinacija sa originalnom slikom:
result = cv2.addWeighted(original_img, 1, lane_on_original, 0.3, 0)


original_image_lane_detection(frame, lanes_img)
Ova funkcija je napravljena takodje kako bismo ovaj program mogli pokrenuti iz fajla za obradu video snimka.

Definisanje matrica perspektivne transformacije:
src = np.float32([[585, 460], [705, 460], [1100, 720], [200, 720]])
dst = np.float32([[300, 0], [1000, 0], [1000, 720], [300, 720]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

Detekcija linija:
left_fit, right_fit = detect_lane_pixels_and_fit_poly(binary_warped)
Ovde pozivamo funkciju iz prethodnog fajla.

Poziv funkcije za preklapanje linija:
final_result = warp_lane_to_original(binary_warped, original_img, left_fit, right_fit, Minv)

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Input video: /test_videos/project_video03.mp4
Output video: /output_videos/project_video_output.avi

#########################
back_to_original_img.py
#########################

U ovom fajlu obradjujemo frejm po frejm i pozivamo sve funkcije koje smo do sad napravili i objasnili u gornjem dijelu koda.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Imam problem sa prebacivanjem u binarnu sliku odredjenih slika iz test_images foldera. Tacnije rade mi samo test.jpg slike.
Iz nekog meni nepoznatog razloga mi vrati samo skroz crnu sliku. To nisam uspio da implementiram. 
Kasnije u daljoj obradi, kod video snimaka takodje imam taj problem. Najbolje se obradi snimak project_video03.mp4 tako da prilazem njegov output.
Nazalost zbog posla nisam uspio da prisustvujem predavanjima ali sam se iz dana u dan trudio da ispratim sve koliko je to moguce preko prezentacija. 
Nazalost, trenutno je ovo neki moj maksimum s obzirom na sve okolnosti oko prisustva. Rijesenje nazalost nije najsrecnije jer ne prolazi kroz sve test slucajeve ali prolazi kroz vecinu.

