import sys
class Lib:
    def __init__(self, listofbooks):
        self.availablebooks = listofbooks

    def DisplayAvailablebooks(self):
        print("Below are the list of books available to be issued")
        print("==================================================")
        for book in self.availablebooks:
            print(book)

    def RequestBook(self, requestedBook):
        if requestedBook in self.availablebooks:
            print("The requested book had been issued")
            self.availablebooks.remove(requestedBook)
        else:
            print("Sorry the book you have requested is already been issued")

    def ReturnBook(self, returnedBook):
        self.availablebooks.append(returnedBook)
        print("Thanks for returning the issued book")


class Stu:
    def __init__(self,listofstudents):
        self.listofstudents = listofstudents
    def requestBook(self):
        print("Enter the name of the book to be issued:")
        self.book = input()
        return self.book

    def returnBook(self):
        print("Please enter the book to be returned: ")
        self.book = input()
        return self.book
class Fac:
    def __init__(self, listoffaculty):  # this init method is the first method to be invoked when you create an object
        # what attributes does a library in general have? - for now, let's abstract and just say it has availablebooks (we're not going to program the shelves, and walls in!)
        self.listoffaculty = listoffaculty



def main():
    booksavailable = Lib(["Game of thrones", "Hobbit", "Two states", "Half Girl friend", "The brown code"])
    faculty = Fac(["Satish","Ramesh", "Suresh"])
    student = Stu(['Srujan',"Alekhya"])
    Name = input("Please Enter your name ")
    if Name in  student.listofstudents or Name in faculty.listoffaculty:
        print(" You are authorised to use the application ")
    else:
        print("Authentication failed")
        sys.exit()



    print(""" ======Library Portal=======
                  1. Available books
                  2. Issue a book
                  3. Return book
                  4. Exit
                  """)
    action = int(input("Choose action :"))
    if action == 1:
            booksavailable.DisplayAvailablebooks()
    elif action == 2:
            booksavailable.DisplayAvailablebooks()
            booksavailable.RequestBook(student.requestBook())
    elif action == 3:
            booksavailable.DisplayAvailablebooks()
            bookname = input("Please enter the book to be returned ")
            if bookname in booksavailable.availablebooks:
                print(" You have given the wrong name. Please retry!!!! ")
            else:

                booksavailable.ReturnBook(student.returnBook())
    elif action == 4:
            sys.exit()


main()


