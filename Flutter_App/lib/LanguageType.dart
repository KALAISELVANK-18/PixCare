import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/UserType.dart';
import 'package:shared_preferences/shared_preferences.dart';
class Language extends StatefulWidget {
  const Language({super.key});

  @override
  State<Language> createState() => _LanguageState();
}

class _LanguageState extends State<Language> {
  @override
  void EnglishUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('userLanguage', 'English');
  }
  void HindiUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('userLanguage', 'hindi');
  }
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
          child: Padding(
            padding: const EdgeInsets.only(top: 300),
            child: Container(
              child: Column(
                children: [
                  Text("Choose Language",style: GoogleFonts.poppins(fontSize: 30),),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [

                      Padding(
                        padding: const EdgeInsets.only(top: 20),
                        child: Container(
                          width: 155,
                          child: OutlinedButton(

                            child: Row(
                              children: [
                                Icon(Icons.language),
                                Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Text("English",style: GoogleFonts.roboto(fontSize: 20,color: Colors.black),),
                                )
                              ],
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            ),

                            onPressed: ()async{
                              EnglishUserType();
                              Navigator.of(context).push(
                                MaterialPageRoute(
                                  builder: (context) => UserType(),
                                ),
                              );
                            },
                          ),

                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.only(top: 20),
                        child: Container(
                          width: 140,
                          child: OutlinedButton(

                            child: Row(
                              children: [
                                Icon(Icons.temple_hindu),
                                Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Text("Hindi",style: GoogleFonts.roboto(fontSize: 20,color: Colors.black),),
                                )
                              ],
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                            ),

                            onPressed: ()async{
                              HindiUserType();
                              Navigator.of(context).push(
                                MaterialPageRoute(
                                  builder: (context) => UserType(),
                                ),
                              );
                            },
                          ),

                        ),
                      ),
                    ],
                  )

                ],
              ),
            ),
          ),
        ),

    );
  }
}
