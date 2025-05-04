import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/main.dart';
import 'package:shared_preferences/shared_preferences.dart';

List<String> list1 = <String>[
  'Light blue,gray,green',
  'Blue, gray or green',
  'Blue',
  'Dark Brown',
  'Brownish Black'
];
List<String> list2 = <String>[
  'Sandy red',
  'Blonde',
  'Chestnut/Dark Blonde',
  'Medium Brown',
  'Dark brown',
  'Black'
];
List<String> list3 = <String>[
  'Reddish',
  'Very Pale',
  'Pale with a beinge tint',
  'Light brown',
  'Dark brown'
];
List<String> list4 = <String>['Many', 'Several', 'Few', 'Incidental', 'None'];
List<String> list5 = <String>[
  'Always burns',
  'Usually burns',
  'Burns sometimes followed by peeling',
  'Rare burns',
  'Never had burns'
];
List<String> list6 = <String>[
  'Hardly or not at all',
  'Sometimes tans',
  'Tans mostly uniformly',
  'Tans more than average',
  'Always tans'
];
List<String> list7 = <String>[
  'Never',
  'Seldom',
  'Sometimes',
  'Often',
  'Always'
];
List<String> list8 = <String>[
  'Very sensistive',
  'Sensitive',
  'Normal',
  'Very resistant',
  'Never had a problem'
];
List<String> list9 = <String>[
  'More than 3 months ago',
  '2-3 months ago',
  '12 months ago',
  'Less than a month ago',
  'Less than 2 weeks ago'
];
List<String> list10 = <String>[
  'Never',
  'Hardly ever',
  'Sometimes',
  'Often',
  'Always'
];

List<String> h1 = <String>[
  'हल्का नीला, धूसर, हरा',
  'नीला, धूसर या हरा',
  'नीला',
  'गहरा भूरा',
  'भूरा-काला'
];
List<String> h2 = <String>[
  'रेतीला लाल',
  'ब्लॉन्ड',
  'चेस्टनट/डार्क ब्लॉन्ड',
  'मध्यम भूरा',
  'गहरा भूरा',
  'काला'
];
List<String> h3 = <String>[
  'रेडिश',
  'बहुत हल्का',
  'हल्का भूरा रंग के साथ',
  'हल्का भूरा',
  'गहरा भूरा'
];
List<String> h4 = <String>['कई', 'कई', 'थोड़े', 'आकस्मिक', 'कोई नहीं'];
List<String> h5 = <String>[
  'हमेशा जलता है',
  'सामान्यत: जलता है',
  'कभी-कभी जलता है, जिसके बाद छिल जाता है',
  'कमी बार-बार जलता है',
  'कभी जलता ही नहीं है'
];
List<String> h6 = <String>[
  'बहुत कम या बिल्कुल नहीं',
  'कभी-कभी टैन होता है',
  'सामान्यत: समृद्धि से टैन होता है',
  'सामान्य से अधिक टैन होता है',
  'हमेशा टैन होता है'
];
List<String> h7 = <String>['कभी नहीं', 'कभी-कभी', 'कभी-कभी', 'अक्सर', 'हमेशा'];
List<String> h8 = <String>[
  'बहुत संवेदनशील',
  'संवेदनशील',
  'सामान्य',
  'बहुत प्रतिरक्षी',
  'कभी समस्या नहीं हुई'
];
List<String> h9 = <String>[
  '3 महीने से अधिक पहले',
  '2-3 महीने पहले',
  '12 महीने पहले',
  'एक महीने से कम पहले',
  '2 सप्ताह से कम पहले'
];
List<String> h10 = <String>['कभी नहीं', 'बहुत कम', 'कभी-कभी', 'अक्सर', 'हमेशा'];

class Patient extends StatefulWidget {
  const Patient({super.key});

  @override
  State<Patient> createState() => _PatientState();
}

class _PatientState extends State<Patient> {
  String dropdown1 = list1.first;
  String dropdown2 = list2.first;
  String dropdown3 = list3.first;
  String dropdown4 = list4.first;
  String dropdown5 = list5.first;
  String dropdown6 = list6.first;
  String dropdown7 = list7.first;
  String dropdown8 = list8.first;
  String dropdown9 = list9.first;
  String dropdown10 = list10.first;
  Future<String?> getUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('userType');
  }

  String? language;

  void initState() {
    super.initState();
    getLanguage();
  }

  Future<String?> getLanguage() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      language = prefs.getString("userLanguage");
    });
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Container(
          child: Column(
            children: [
              Container(
                height: MediaQuery.of(context).size.height * 0.3,
                width: MediaQuery.of(context).size.width,
                child: Padding(
                  padding:
                  const EdgeInsets.only(top: 100.0, left: 10, right: 10),
                  child: Column(
                    children: [
                      Text(
                        language == 'hindi'
                            ? "आपके त्वचा के प्रकार की पहचान"
                            : "Identifying your skin type",
                        style: GoogleFonts.poppins(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top: 12.0),
                        child: Text(
                          language == 'hindi'
                              ? "अगर आपने कभी सही फाउंडेशन ढूंढ़ने का प्रयास किया है, तो"
                              : "If you have ever tried to find the perfect foundation, you",
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: Colors.white,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                      Text(
                        language == 'hindi'
                            ? "हो सकता है कि आपने फिटजपैट्रिक परीक्षण के बारे में सुना हो।"
                            : "might have heard of the Fitzpatrick test.",
                        style: GoogleFonts.poppins(
                          fontSize: 12,
                          color: Colors.white,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
                decoration: BoxDecoration(
                    color: Color.fromRGBO(86, 128, 227, 1),
                    borderRadius: BorderRadius.only(
                      bottomLeft: Radius.circular(20),
                      bottomRight: Radius.circular(20),
                    )),
              ),
              Padding(
                padding: const EdgeInsets.only(left: 20, top: 20, right: 20),
                child: Container(
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Text(
                              language == 'hindi'
                                  ? "आपकी आंखों का रंग क्या है?"
                                  : "What color are your eyes?",
                              style: GoogleFonts.poppins(
                                fontSize: 15,
                              ),
                            ),
                          ],
                          mainAxisAlignment: MainAxisAlignment.start,
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h1.first : list1.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h1.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list1
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "आपका प्राकृतिक बाल का रंग क्या है?"
                                      : "What is your natural hair color?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h2.first : list2.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h2.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list2
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "आपका त्वचा का रंग (अनुभूत क्षेत्र) क्या है?"
                                      : "What is your skin color (unexposed areas)?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h3.first : list3.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h3.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list3
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "क्या आपके अनुभूत क्षेत्रों पर तिल (freckles) हैं?"
                                      : "Do you have freckles on unexposed areas?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h4.first : list4.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h4.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list4
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "आप सूर्य में बहुत देर रहने पर क्या होता है?"
                                      : "What happens when you stay too long in the sun?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h5.first : list5.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h5.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list5
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "आप किस मात्रा में भूरा हो जाते हैं?"
                                      : "To what degree do you turn brown?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h6.first : list6.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h6.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list6
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "क्या आप सूर्य प्रकाश के कई घंटों के बाद भूरे हो जाते हैं?"
                                      : "Do you turn brown after several hours of sun exposure?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h7.first : list7.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h7.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list7
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "सूर्य के प्रति आपका चेहरा कैसे प्रतिक्रिया करता है?"
                                      : "How does your face react to the sun?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h8.first : list8.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h8.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list8
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "आपने अपने शरीर को आखिरकार सूर्य (या कृत्रिम सूर्य लैम्प / टैनिंग क्रीम) के सामने कब रखा था?"
                                      : "When did you last expose your body to the sun (or artificial sunlamp/tanning cream)?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h9.first : list9.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h9.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list9
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(top: 10.0),
                          child: Row(
                            children: [
                              Container(
                                width: MediaQuery.of(context).size.width * 0.8955,
                                child: Text(
                                  language == 'hindi'
                                      ? "क्या आपने उपचार के लिए क्षेत्र को सूर्य के सामने किया था?"
                                      : "Did you expose the area to be treated to the sun?",
                                  style: GoogleFonts.poppins(
                                    fontSize: 15,
                                  ),
                                ),
                              ),
                            ],
                            mainAxisAlignment: MainAxisAlignment.start,
                          ),
                        ),
                        DropdownMenu<String>(
                          initialSelection:
                          language == 'hindi' ? h10.first : list10.first,
                          width: MediaQuery.of(context).size.width * 0.90,
                          onSelected: (String? value) {
                            // This is called when the user selects an item.
                            setState(() {
                              dropdown1 = value!;
                            });
                          },
                          dropdownMenuEntries: language == 'hindi'
                              ? h10.map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList()
                              : list10
                              .map<DropdownMenuEntry<String>>((String value) {
                            return DropdownMenuEntry<String>(
                                value: value, label: value);
                          }).toList(),
                        ),
                      ],
                    )),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 20.0, bottom: 20),
                child: Container(
                  width: MediaQuery.of(context).size.width * 0.9,
                  decoration: BoxDecoration(
                      color: Color.fromRGBO(86, 128, 227, 1),
                      borderRadius: BorderRadius.circular(20)),
                  child: TextButton(
                    onPressed: () {
                      String type;
                      if (dropdown2 == "Blonde" &&
                          dropdown1 == "Bllue, gray or green" &&
                          dropdown3 == "Very Pale" &&
                          dropdown5 == "Always burns" &&
                          dropdown6 == "Hardly or not at all") {
                        type = "Type I";
                        _showMyDialog("images/img_1.png", type, context);
                      } else if (dropdown2 == "Sandy red" &&
                          dropdown1 == "Blue" &&
                          dropdown3 == "Pale with a beinge tint" &&
                          dropdown5 == "Usually burns" &&
                          dropdown6 == "Sometimes tans") {
                        type = "Type II";
                        _showMyDialog("images/img_2.png", type, context);
                      } else if (dropdown2 == "Chestnut/Dark blonde" &&
                          dropdown1 == "Dark Brown" &&
                          dropdown3 == "Reddish" &&
                          dropdown5 == "Burns sometimes followed by peeling" &&
                          dropdown6 == "Tans mostly uniformly") {
                        type = "Type III";
                        _showMyDialog("images/img_3.png", type, context);
                      } else if (dropdown2 == "Medium Brown" &&
                          dropdown1 == "Dark Brown" &&
                          dropdown3 == "Light brown" &&
                          dropdown5 == "Rare burns" &&
                          dropdown6 == "Always tans") {
                        type = "Type IV";
                        _showMyDialog("images/img_4.png", type, context);
                      } else if (dropdown2 == "Dark brown" &&
                          dropdown1 == "Dark Brown" &&
                          dropdown3 == "Dark brown" &&
                          dropdown5 == "Rare burns" &&
                          dropdown6 == "Tans more than average") {
                        type = "Type V";
                        _showMyDialog("images/img_5.png", type, context);
                      } else {
                        type = "Type VI";
                        _showMyDialog("images/img_6.png", type, context);
                      }
                    },
                    child: Text(
                      language == 'hindi' ? "अगला कदम" : "Next Step",
                      style: GoogleFonts.poppins(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _showMyDialog(
      String image, String type, BuildContext context) async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false, // user must tap button!
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(
            language == 'hindi'
                ? 'आपकी त्वचा का प्रकार: $type'
                : 'Your Skin type: $type',
          ),
          content: SingleChildScrollView(
            child: Container(
              child: ListBody(
                children: <Widget>[
                  Image.asset(
                    image,
                    height: 200,
                    width: 100,
                    fit: BoxFit.cover,
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Container(
                      decoration: BoxDecoration(
                          color: Color.fromRGBO(86, 128, 227, 1),
                          borderRadius: BorderRadius.circular(20)),
                      child: TextButton(
                        onPressed: () async {
                          String? usertype = await getUserType();
                          if (usertype != null)
                            await FirebaseFirestore.instance
                                .collection(usertype)
                                .doc(FirebaseAuth.instance.currentUser!.email)
                                .set({
                              "Type ": type,
                              "Head(सिर)": 0,
                              "Body(शरीर)": 0,
                              "Left Hand(बाएं हाथ)": 0,
                              "Right Hand(दाएं हाथ)": 0,
                              "Left Leg(बाएं पैर)": 0,
                              "Right Leg(दाएं पैर)": 0,
                              "userid":FirebaseAuth.instance.currentUser!.email
                            });

                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (context) => HomePage(),
                            ),
                          );
                        },
                        child: Text(
                          "Next",
                          style: GoogleFonts.poppins(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}